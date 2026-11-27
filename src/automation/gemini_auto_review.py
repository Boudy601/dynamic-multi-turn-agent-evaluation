#!/usr/bin/env python3
"""Automate SOP review and repair using Gemini Web via Botasaurus."""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_GEMINI_URL = "https://gemini.google.com/app"
DEFAULT_PROFILE = "gemini_reviewer"
DEFAULT_MODEL_NAME = "Neon"
DEFAULT_THINKING_EFFORT = "Low"
DEFAULT_AUTO_ADD_COUNT = 5
MAX_REASON_LENGTH = 300
VALID_SPEAKERS = {"agent", "customer", "action"}


class AutomationError(RuntimeError):
    """Raised when server communication or browser automation fails."""


@dataclass(frozen=True)
class ParsedResponse:
    outcome: str  # "edited", "no_error", or "ambiguous"
    summary: str
    messages: list[dict[str, Any]] | None
    raw_text: str


def build_evaluation_prompt(chat: Mapping[str, Any]) -> str:
    """Build the exact SOP evaluation + simulator-repair prompt used in the UI."""
    formatted_turns: list[str] = []
    for index, message in enumerate(chat.get("messages", []), start=1):
        speaker = str(message.get("speaker", "unknown"))
        text = str(message.get("text", ""))
        header = f"{index}. [{speaker.upper()}] {text}"
        if speaker.casefold() != "action":
            formatted_turns.append(header)
            continue
        raw_params = message.get("parameters")
        params = (
            ", ".join(str(p) for p in raw_params)
            if isinstance(raw_params, list) and raw_params
            else "(none)"
        )
        raw_expected = message.get("expected_parameters")
        if not isinstance(raw_expected, list):
            expected = "Not listed for this subflow"
        elif raw_expected:
            expected = ", ".join(str(p) for p in raw_expected)
        else:
            expected = "(none)"
        fn_name = str(message.get("function_name") or "unknown")
        formatted_turns.append(
            f"{header}\n   Function: {fn_name}\n   Parameters: {params}\n   Expected parameters: {expected}"
        )

    chat_lines = "\n\n".join(formatted_turns)
    sop_name = str(chat.get("sop_name", "SOP"))
    sop_json = json.dumps(chat.get("sop_guideline") or {}, ensure_ascii=False, indent=2)
    scenario_json = json.dumps(chat.get("scenario") or {}, ensure_ascii=False, indent=2)

    return "\n".join([
        "Check this chat against the attached SOP. Treat the SOP as the ground truth—note that the Expected parameters: lines in the chat logs are just system hints showing possible parameters, not strict SOP requirements.",
        "",
        "Important evaluation rules:",
        "- Use practical, human common sense. Do NOT nitpick natural conversational flow, small talk, empathy, or minor step-ordering flexibility (e.g., chatting briefly or reassuring the customer before asking for their name/Account ID is completely fine).",
        "- Focus ONLY on meaningful SOP violations, such as: skipping required tool calls/actions, entering wrong or blank parameters, following the wrong workflow/resolution, failing to provide required information/links/codes, or breaking explicit business rules (like issuing a refund when they shouldn't).",
        "",
        "If the chat contains violations, you will also fix the chat:",
        "- Keep all valid turns before the first mistake intact.",
        "- Fix the mistaken turn(s) naturally without changing the tone, style, or dynamic of the original conversation.",
        "- If an early mistake (e.g., calling the wrong tool, entering wrong parameters, or taking the wrong workflow path) causes downstream propagation errors, act as a simulator for both the customer and the agent from that point forward—re-simulating the remaining turns naturally so everything stays consistent with the User Scenario and the SOP through resolution.",
        "",
        "Format your response exactly like this:",
        "",
        "Violations",
        '(If there are none, state "NO/No major SOP violations.")',
        "",
        "[Step Name]: [Brief note on what the agent did correctly or violated]",
        "",
        "~300-Character Summary",
        "[A concise ~300-character summary of the violations, or a confirmation that there were none.]",
        "",
        "Corrected Chat (JSON)",
        '(Omit this section if there are no major SOP violations. Otherwise, output ONLY a single ```json code block containing the full corrected conversation array. Each item must have "speaker" ("agent", "customer", or "action") and "text". For "action" items, also include "function_name" (string) and "parameters" (array of strings).)',
        "",
        "---",
        f"SOP ({sop_name}):",
        sop_json,
        "",
        "---",
        "User Scenario:",
        scenario_json,
        "",
        "---",
        "Chat:",
        chat_lines,
    ])


def normalize_message_list(raw_list: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_list, list) or not raw_list:
        return None
    normalized: list[dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, Mapping):
            return None
        speaker_raw = item.get("speaker")
        if not isinstance(speaker_raw, str):
            return None
        speaker = speaker_raw.strip().casefold()
        if speaker not in VALID_SPEAKERS:
            return None
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        entry: dict[str, Any] = {"speaker": speaker, "text": text}
        if speaker == "action":
            fn_name = item.get("function_name", "")
            params = item.get("parameters", [])
            if fn_name is None:
                fn_name = ""
            if not isinstance(fn_name, str):
                return None
            if params is None:
                params = []
            if not isinstance(params, list) or any(not isinstance(p, str) for p in params):
                return None
            entry["function_name"] = fn_name.strip()
            entry["parameters"] = list(params)
        normalized.append(entry)
    return normalized


def extract_corrected_messages(
    raw_text: str, code_blocks: Sequence[str] = ()
) -> list[dict[str, Any]] | None:
    candidates: list[str] = []
    for block in code_blocks:
        cleaned = block.strip()
        if cleaned:
            candidates.append(cleaned)

    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text, re.IGNORECASE):
        candidates.append(match.group(1).strip())

    # Fallback: scan JSON arrays in raw_text from last to first (in case Gemini rendered without fences)
    decoder = json.JSONDecoder()
    for idx in range(len(raw_text) - 1, -1, -1):
        if raw_text[idx] == "[":
            try:
                value, _ = decoder.raw_decode(raw_text[idx:])
            except json.JSONDecodeError:
                continue
            normalized = normalize_message_list(value)
            if normalized is not None:
                return normalized

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and "messages" in value:
            value = value["messages"]
        normalized = normalize_message_list(value)
        if normalized is not None:
            return normalized
    return None


def cap_reason(text: str, limit: int = MAX_REASON_LENGTH) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip().strip('"').strip("'")
    if len(cleaned) <= limit:
        return cleaned
    cutoff = cleaned[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    if not cutoff:
        cutoff = cleaned[: limit - 1]
    return cutoff + "…"


def extract_summary_reason(raw_text: str) -> str:
    heading_pattern = re.compile(
        r"~?\s*300[-\s]*Character\s+Summary\s*:?\s*\n*\s*([\s\S]+?)(?=\s*(?:#+\s*)?(?:\*\*)?Corrected\s+Chat|\n\s*```|\n\s*---|\s*```|$)",
        re.IGNORECASE,
    )
    match = heading_pattern.search(raw_text)
    if match:
        summary_block = match.group(1).strip()
        # Strip wrapping brackets if the model echoed [ ... ]
        if summary_block.startswith("[") and summary_block.endswith("]"):
            summary_block = summary_block[1:-1].strip()
        if summary_block:
            return cap_reason(summary_block)

    # Fallback: grab Violations section if summary heading was formatted differently
    violations_match = re.search(
        r"Violations\s*:?\s*\n*\s*([\s\S]+?)(?=\s*(?:#+\s*)?(?:\*\*)?(?:~?\s*300[-\s]*Character\s+Summary|Corrected\s+Chat)|$)",
        raw_text,
        re.IGNORECASE,
    )
    if violations_match:
        lines = [
            line.strip(" -*")
            for line in violations_match.group(1).splitlines()
            if line.strip()
        ]
        if lines:
            return cap_reason(" ".join(lines))
    return ""


def parse_gemini_response(
    raw_text: str, code_blocks: Sequence[str] = ()
) -> ParsedResponse:
    messages = extract_corrected_messages(raw_text, code_blocks)
    summary = extract_summary_reason(raw_text)

    if messages is not None:
        if not summary:
            summary = "Edited by Gemini due to SOP violation."
        return ParsedResponse(
            outcome="edited",
            summary=summary,
            messages=messages,
            raw_text=raw_text,
        )

    lowered = raw_text.casefold()
    no_violations = (
        "no major sop violations" in lowered
        or "no/no major sop violations" in lowered
        or "no sop violations" in lowered
    )
    if no_violations:
        return ParsedResponse(
            outcome="no_error",
            summary=summary,
            messages=None,
            raw_text=raw_text,
        )

    return ParsedResponse(
        outcome="ambiguous",
        summary=summary,
        messages=None,
        raw_text=raw_text,
    )


class ReviewServerClient:
    """Thin HTTP client for server.py endpoints."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _request_json(
        self, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if payload is None:
            req = urllib.request.Request(url, method="GET")
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AutomationError(
                f"Server HTTP {exc.code} on {path}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise AutomationError(
                f"Cannot reach review server at {self.base_url} ({exc.reason}). "
                "Start it first with: python server.py"
            ) from exc

    def get_chats(self, reviewer: str) -> list[dict[str, Any]]:
        encoded = urllib.parse.quote(reviewer, safe="")
        data = self._request_json(f"/api/chats/{encoded}")
        return list(data.get("chats", []))

    def add_chats(self, reviewer: str, count: int) -> dict[str, Any]:
        return self._request_json(
            "/api/add-chats",
            {"reviewer": reviewer, "count": count},
        )

    def update_chat(
        self, reviewer: str, chat_id: str, messages: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        return self._request_json(
            "/api/update-chat",
            {"reviewer": reviewer, "chat_id": chat_id, "messages": list(messages)},
        )

    def save_review(
        self,
        reviewer: str,
        chat_id: str,
        status: str,
        reason: str = "",
        edited: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "reviewer": reviewer,
            "chat_id": chat_id,
            "status": status,
            "reviewed_by": "ai",
        }
        if reason:
            body["reason"] = reason
        if edited:
            body["edited"] = True
        return self._request_json("/api/review", body)


_FIND_INPUT_JS = """
const selectors = [
  'rich-textarea div[contenteditable="true"]',
  'div.ql-editor[contenteditable="true"]',
  'div[contenteditable="true"][role="textbox"]',
  'textarea[aria-label*="prompt" i]',
  'textarea'
];
for (const sel of selectors) {
  const el = document.querySelector(sel);
  if (el && el.offsetParent !== null) return true;
}
return false;
"""

_NEEDS_SIGNIN_JS = """
const url = window.location.href || '';
if (url.includes('accounts.google.com')) return true;
const signIn = Array.from(document.querySelectorAll('a, button')).find(el => {
  if (!el || el.offsetParent === null) return false;
  const t = (el.innerText || el.getAttribute('aria-label') || '').trim().toLowerCase();
  return t === 'sign in' || t === 'sign in to gemini';
});
return Boolean(signIn);
"""

_SET_PROMPT_JS = """
const bytes = Uint8Array.from(atob(args.b64), c => c.charCodeAt(0));
const promptText = new TextDecoder('utf-8').decode(bytes);
const selectors = [
  'rich-textarea div[contenteditable="true"]',
  'div.ql-editor[contenteditable="true"]',
  'div[contenteditable="true"][role="textbox"]',
  'textarea[aria-label*="prompt" i]',
  'textarea'
];
let el = null;
for (const sel of selectors) {
  const cand = document.querySelector(sel);
  if (cand && cand.offsetParent !== null) { el = cand; break; }
}
if (!el) return { ok: false, error: 'Prompt input element not found' };
el.focus();
if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
  el.value = promptText;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  return { ok: true, method: 'value' };
}
try {
  const range = document.createRange();
  range.selectNodeContents(el);
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
  const inserted = document.execCommand('insertText', false, promptText);
  if (inserted && (el.innerText || '').trim().length > 0) {
    return { ok: true, method: 'execCommand' };
  }
} catch (_) {}
el.textContent = promptText;
el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: promptText }));
return { ok: true, method: 'textContent' };
"""

_CLICK_NEW_CHAT_JS = """
const selectors = [
  'button[aria-label*="New chat" i]',
  'a[aria-label*="New chat" i]',
  'a[href="/app"]',
  '[data-test-id="new-chat-button"] button',
  '[data-test-id="new-chat-button"]'
];
for (const sel of selectors) {
  const el = document.querySelector(sel);
  if (el && el.offsetParent !== null && !el.disabled && el.getAttribute('aria-disabled') !== 'true') {
    return sel;
  }
}
return null;
"""

_FIND_SEND_SELECTOR_JS = """
const selectors = [
  'button[aria-label*="Send" i]',
  'button.send-button',
  'button[data-test-id="send-button"]'
];
for (const sel of selectors) {
  const el = document.querySelector(sel);
  if (el && !el.disabled && el.getAttribute('aria-disabled') !== 'true' && el.offsetParent !== null) {
    return sel;
  }
}
return null;
"""

_CLICK_SEND_JS = """
const selectors = [
  'button[aria-label*="Send" i]',
  'button.send-button',
  'button[data-test-id="send-button"]',
  'mat-icon[data-mat-icon-name="send"]'
];
for (const sel of selectors) {
  let el = document.querySelector(sel);
  if (el && el.tagName !== 'BUTTON') el = el.closest('button');
  if (el && !el.disabled && el.getAttribute('aria-disabled') !== 'true' && el.offsetParent !== null) {
    el.click();
    return true;
  }
}
return false;
"""

_RESPONSE_STATE_JS = """
const responseEls = Array.from(document.querySelectorAll(
  'model-response, .model-response-container, message-content, [data-message-author-role="model"]'
)).filter(el => el.offsetParent !== null);

// Prefer outer model-response containers if present so message-content isn't double counted
const outer = Array.from(document.querySelectorAll('model-response')).filter(el => el.offsetParent !== null);
const list = outer.length ? outer : responseEls;

const stopBtn = Array.from(document.querySelectorAll(
  'button[aria-label*="Stop" i], button[mattooltip*="Stop" i]'
)).find(b => b.offsetParent !== null);

const last = list.length ? list[list.length - 1] : null;
let rawText = '';
let codeBlocks = [];
if (last) {
  const clone = last.cloneNode(true);
  clone.querySelectorAll('button, .code-block-decoration, .header-formatted').forEach(n => n.remove());
  rawText = (clone.innerText || '').trim();
  codeBlocks = Array.from(last.querySelectorAll('code-block pre, pre code, code-block code')).map(
    n => (n.innerText || '').trim()
  ).filter(Boolean);
}
return {
  count: list.length,
  isGenerating: Boolean(stopBtn),
  rawText,
  codeBlocks
};
"""

_SELECT_MODEL_JS = """
const targetModel = (args.model || '').toLowerCase();
const targetThinking = (args.thinking || '').toLowerCase();
const isVisible = el => el && el.offsetParent !== null;

const switcherSelectors = [
  'bard-mode-switcher button',
  'button[data-test-id*="bard-mode" i]',
  'button[aria-label*="model" i]',
  'button[aria-label*="mode" i]',
  '.logo-pill-btn',
  'button.mdc-button'
];

let switcher = null;
for (const sel of switcherSelectors) {
  for (const el of document.querySelectorAll(sel)) {
    if (!isVisible(el)) continue;
    const txt = (el.innerText || el.getAttribute('aria-label') || '').toLowerCase();
    if (
      sel.includes('bard-mode') ||
      (targetModel && txt.includes(targetModel)) ||
      txt.includes('neon') ||
      txt.includes('hydrogen') ||
      txt.includes('gemini') ||
      txt.includes('pro') ||
      txt.includes('flash') ||
      txt.includes('thinking')
    ) {
      switcher = el;
      break;
    }
  }
  if (switcher) break;
}

if (!switcher) {
  return { status: 'not_found', current: '' };
}
const currentText = (switcher.innerText || '').trim();
if (targetModel && currentText.toLowerCase().includes(targetModel) && (!targetThinking || currentText.toLowerCase().includes(targetThinking))) {
  return { status: 'already_selected', current: currentText };
}
switcher.click();
return { status: 'opened', current: currentText };
"""

_PICK_MODEL_OPTION_JS = """
const targetModel = (args.model || '').toLowerCase();
const targetThinking = (args.thinking || '').toLowerCase();
const isVisible = el => el && el.offsetParent !== null;

const items = Array.from(document.querySelectorAll(
  '[role="menuitem"], [role="option"], [role="menuitemradio"], .mat-mdc-menu-item, button'
)).filter(isVisible);

let clickedModel = false;
if (targetModel) {
  const match = items.find(el => (el.innerText || '').toLowerCase().includes(targetModel));
  if (match) {
    match.click();
    clickedModel = true;
  }
}
let clickedThinking = false;
if (targetThinking) {
  const tMatch = items.find(el => {
    const t = (el.innerText || '').trim().toLowerCase();
    return t === targetThinking || (t.includes(targetThinking) && (t.includes('thinking') || t.includes('effort') || t.startsWith(targetThinking)));
  });
  if (tMatch) {
    tMatch.click();
    clickedThinking = true;
  }
}
return { clickedModel, clickedThinking };
"""

_OPEN_THINKING_PICKER_JS = """
const isVisible = el => el && el.offsetParent !== null;
const levels = ['minimal', 'low', 'medium', 'high'];
const btn = Array.from(document.querySelectorAll('button, [role="button"]')).find(el => {
  if (!isVisible(el)) return false;
  const label = (el.getAttribute('aria-label') || '').toLowerCase();
  const txt = (el.innerText || '').trim().toLowerCase();
  return label.includes('thinking') || label.includes('effort') || txt.includes('thinking') || levels.includes(txt);
});
if (!btn) return false;
btn.click();
return true;
"""

_OPEN_CHAT_MENU_JS = """
const isVisible = el => el && el.offsetParent !== null;

// 1. Try top-bar conversation actions menu if present
const topMenuSelectors = [
  'button[aria-label*="Conversation options" i]',
  'button[aria-label*="More options for" i]',
  'conversation-actions button',
  '[data-test-id="conversation-actions-menu-button"]'
];
for (const sel of topMenuSelectors) {
  const btn = document.querySelector(sel);
  if (isVisible(btn)) {
    btn.click();
    return { opened: true, source: 'top' };
  }
}

// 2. Find active conversation in sidebar (open sidebar first if needed)
const findActiveItem = () => {
  const path = window.location.pathname || '';
  const slug = path.split('/').filter(Boolean).pop() || '';
  if (slug && slug !== 'app') {
    const byHref = Array.from(document.querySelectorAll('a[href*="/app/"]')).find(
      a => (a.getAttribute('href') || '').includes(slug) && isVisible(a)
    );
    if (byHref) return byHref.closest('[data-test-id="conversation"], .conversation, li, div.conversation-items-container') || byHref;
  }
  const activeSelectors = [
    '[data-test-id="conversation"].selected',
    '.conversation.selected',
    'a[aria-current="page"]',
    '[aria-current="page"]'
  ];
  for (const sel of activeSelectors) {
    const el = document.querySelector(sel);
    if (isVisible(el)) {
      return el.closest('[data-test-id="conversation"], .conversation, li') || el;
    }
  }
  return null;
};

let activeItem = findActiveItem();
if (!activeItem) {
  const navToggle = document.querySelector(
    'button[aria-label*="Main menu" i], button[aria-label*="Expand" i], button[data-test-id="side-nav-menu-button"]'
  );
  if (isVisible(navToggle)) {
    navToggle.click();
    return { opened: false, sidebarToggled: true };
  }
  return { opened: false, reason: 'no_active_item' };
}

activeItem.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
activeItem.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
const container = activeItem.parentElement || activeItem;
const menuBtn =
  activeItem.querySelector('button[data-test-id="actions-menu-button"], button[aria-label*="More" i], button[aria-label*="options" i]') ||
  container.querySelector('button[data-test-id="actions-menu-button"], button[aria-label*="More" i], button[aria-label*="options" i]');
if (menuBtn) {
  menuBtn.click();
  return { opened: true, source: 'sidebar' };
}
return { opened: false, reason: 'no_menu_button' };
"""

_CLICK_DELETE_MENU_ITEM_JS = """
const isVisible = el => el && el.offsetParent !== null;
const items = Array.from(document.querySelectorAll(
  '[role="menuitem"], .mat-mdc-menu-item, [role="menu"] button'
)).filter(isVisible);
const del = items.find(el => {
  const t = (el.innerText || el.getAttribute('aria-label') || '').trim().toLowerCase();
  return t === 'delete' || t.startsWith('delete ') || t.includes('delete chat');
});
if (!del) return false;
del.click();
return true;
"""

_CONFIRM_DELETE_DIALOG_JS = """
const isVisible = el => el && el.offsetParent !== null;
const dialog = document.querySelector('[role="dialog"], .mat-mdc-dialog-container, mat-dialog-container');
if (!dialog || !isVisible(dialog)) return false;
const buttons = Array.from(dialog.querySelectorAll('button')).filter(isVisible);
const confirmBtn = buttons.find(btn => {
  const t = (btn.innerText || btn.getAttribute('aria-label') || '').trim().toLowerCase();
  const testId = (btn.getAttribute('data-test-id') || '').toLowerCase();
  return t === 'delete' || testId.includes('confirm');
});
if (!confirmBtn) return false;
confirmBtn.click();
return true;
"""


def load_cookies_file(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutomationError(f"Unable to read cookies file {path}: {exc}") from exc
    if not isinstance(raw, list) or any(not isinstance(c, Mapping) for c in raw):
        raise AutomationError(f"Cookies file {path} must contain a JSON array of cookie objects.")
    cleaned: list[dict[str, Any]] = []
    for cookie in raw:
        item: dict[str, Any] = {
            "name": str(cookie.get("name", "")),
            "value": str(cookie.get("value", "")),
        }
        if not item["name"]:
            continue
        for key in ("domain", "path", "secure", "httpOnly", "sameSite", "url"):
            if key in cookie and cookie[key] is not None:
                item[key] = cookie[key]
        if "expirationDate" in cookie and isinstance(cookie["expirationDate"], (int, float)):
            item["expires"] = int(cookie["expirationDate"])
        elif "expires" in cookie and isinstance(cookie["expires"], (int, float)):
            item["expires"] = int(cookie["expires"])
        cleaned.append(item)
    return cleaned


def human_click_selector(driver: Any, selector: str | None) -> bool:
    if not selector:
        return False
    el = driver.select(selector, wait=2)
    if el is None:
        return False
    el.click()
    return True


def ensure_gemini_ready(
    driver: Any,
    gemini_url: str,
    model_name: str,
    thinking_effort: str,
    confirm_model: bool,
    is_first_chat: bool,
    already_on_new_chat: bool = False,
    cookies: Sequence[Mapping[str, Any]] = (),
) -> None:
    if is_first_chat:
        if cookies:
            driver.google_get("https://gemini.google.com/")
            driver.add_cookies([dict(c) for c in cookies])
            driver.get_via_this_page(gemini_url)
        else:
            driver.google_get(gemini_url)
    elif not already_on_new_chat:
        new_chat_sel = driver.run_js(_CLICK_NEW_CHAT_JS)
        if not human_click_selector(driver, new_chat_sel):
            driver.get_via_this_page(gemini_url)

    driver.short_random_sleep()
    for _ in range(16):
        if driver.run_js(_NEEDS_SIGNIN_JS):
            break
        if driver.run_js(_FIND_INPUT_JS):
            break
        time.sleep(0.5)

    if driver.run_js(_NEEDS_SIGNIN_JS) or not driver.run_js(_FIND_INPUT_JS):
        driver.prompt(
            "Please sign into your Google account in the opened Chrome window and open Gemini, then press Enter here..."
        )
        for _ in range(20):
            if not driver.run_js(_NEEDS_SIGNIN_JS) and driver.run_js(_FIND_INPUT_JS):
                break
            time.sleep(0.5)
        if not driver.run_js(_FIND_INPUT_JS):
            raise AutomationError("Gemini prompt input did not become ready.")

    if model_name and is_first_chat:
        res = driver.run_js(
            _SELECT_MODEL_JS,
            {"model": model_name, "thinking": thinking_effort},
        )
        status = (res or {}).get("status")
        model_ok = status == "already_selected"
        thinking_ok = not thinking_effort or status == "already_selected"
        if status == "opened":
            time.sleep(random.uniform(0.5, 1.0))
            pick = driver.run_js(
                _PICK_MODEL_OPTION_JS,
                {"model": model_name, "thinking": thinking_effort},
            )
            model_ok = bool((pick or {}).get("clickedModel"))
            thinking_ok = not thinking_effort or bool((pick or {}).get("clickedThinking"))
            time.sleep(random.uniform(0.4, 0.8))
            if thinking_effort and not thinking_ok:
                if driver.run_js(_OPEN_THINKING_PICKER_JS):
                    time.sleep(random.uniform(0.4, 0.7))
                retry_pick = driver.run_js(
                    _PICK_MODEL_OPTION_JS,
                    {"model": "", "thinking": thinking_effort},
                )
                thinking_ok = bool((retry_pick or {}).get("clickedThinking"))
        if (not model_ok or not thinking_ok) and confirm_model:
            driver.prompt(
                f"Verify/select model '{model_name}'"
                + (f" ({thinking_effort} thinking)" if thinking_effort else "")
                + " in Chrome, then press Enter..."
            )


def submit_and_wait_response(
    driver: Any, prompt_text: str, timeout_seconds: int
) -> tuple[str, list[str]]:
    before = driver.run_js(_RESPONSE_STATE_JS) or {"count": 0}
    initial_count = int(before.get("count", 0))

    prompt_b64 = base64.b64encode(prompt_text.encode("utf-8")).decode("ascii")
    inject_res = driver.run_js(_SET_PROMPT_JS, {"b64": prompt_b64})
    if not (inject_res or {}).get("ok"):
        raise AutomationError(f"Failed to set prompt in Gemini: {inject_res}")

    time.sleep(random.uniform(1.0, 2.2))
    send_sel = driver.run_js(_FIND_SEND_SELECTOR_JS)
    if not human_click_selector(driver, send_sel) and not driver.run_js(_CLICK_SEND_JS):
        raise AutomationError("Could not find or click Gemini Send button.")

    deadline = time.monotonic() + timeout_seconds
    stable_ticks = 0
    last_text = ""
    last_blocks: list[str] = []

    while time.monotonic() < deadline:
        time.sleep(1.5)
        state = driver.run_js(_RESPONSE_STATE_JS) or {}
        count = int(state.get("count", 0))
        is_generating = bool(state.get("isGenerating", False))
        raw_text = str(state.get("rawText", "")).strip()
        code_blocks = list(state.get("codeBlocks") or [])

        if count <= initial_count or not raw_text:
            stable_ticks = 0
            continue

        if is_generating:
            stable_ticks = 0
            last_text = raw_text
            last_blocks = code_blocks
            continue

        if raw_text == last_text:
            stable_ticks += 1
        else:
            stable_ticks = 0
            last_text = raw_text
            last_blocks = code_blocks

        if stable_ticks >= 2:
            return last_text, last_blocks

    raise AutomationError(
        f"Timed out after {timeout_seconds}s waiting for Gemini response."
    )


def write_log_entry(log_path: Path, entry: Mapping[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def delete_current_gemini_chat(driver: Any) -> bool:
    """Best-effort delete of the active Gemini conversation from the sidebar."""
    opened = driver.run_js(_OPEN_CHAT_MENU_JS) or {}
    if opened.get("sidebarToggled"):
        time.sleep(0.5)
        opened = driver.run_js(_OPEN_CHAT_MENU_JS) or {}
    if not opened.get("opened"):
        return False

    time.sleep(0.4)
    if not driver.run_js(_CLICK_DELETE_MENU_ITEM_JS):
        return False

    for _ in range(8):
        time.sleep(0.3)
        if driver.run_js(_CONFIRM_DELETE_DIALOG_JS):
            time.sleep(0.6)
            return True
    return False


def run_batch(args: argparse.Namespace) -> int:
    try:
        from botasaurus.browser import Driver, browser
    except ImportError as exc:
        print(
            "Error: botasaurus is not installed. Run: pip install botasaurus",
            file=sys.stderr,
        )
        return 2

    client = ReviewServerClient(args.server_url)
    if args.add_count is not None:
        if args.add_count < 1:
            raise AutomationError("--add-count must be a positive integer.")
        if args.dry_run:
            print("Dry-run enabled: skipping /api/add-chats.")
        else:
            add_res = client.add_chats(args.reviewer, args.add_count)
            added = int(add_res.get("added", 0))
            print(f"Added {added} new chat(s) to '{args.reviewer}'.")
            if add_res.get("warning"):
                print(f"Warning: {add_res['warning']}", file=sys.stderr)

    chats = client.get_chats(args.reviewer)
    target_statuses = {"pending"}
    if args.include_error:
        target_statuses.add("error")

    def build_queue(all_chats: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = [c for c in all_chats if c.get("status") in target_statuses]
        if args.chat_id:
            selected = [c for c in selected if c.get("chat_id") == args.chat_id]
        if args.limit is not None:
            selected = selected[: args.limit]
        return selected

    queue = build_queue(chats)
    if (
        not queue
        and args.add_count is None
        and not args.chat_id
        and not args.dry_run
    ):
        auto_add = (
            args.limit
            if args.limit is not None and args.limit > 0
            else DEFAULT_AUTO_ADD_COUNT
        )
        print(
            f"No pending chats found for '{args.reviewer}'. "
            f"Auto-adding {auto_add} chat(s)..."
        )
        add_res = client.add_chats(args.reviewer, auto_add)
        added = int(add_res.get("added", 0))
        print(f"Added {added} new chat(s) to '{args.reviewer}'.")
        if add_res.get("warning"):
            print(f"Warning: {add_res['warning']}", file=sys.stderr)
        chats = client.get_chats(args.reviewer)
        queue = build_queue(chats)

    if not queue:
        print(f"No matching pending chats found for reviewer '{args.reviewer}'.")
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = (
        PROJECT_ROOT / "logs" / f"gemini_auto_review_{args.reviewer}_{timestamp}.jsonl"
    )
    print(
        f"Processing {len(queue)} chat(s) for '{args.reviewer}' "
        f"(model='{args.model}', thinking='{args.thinking}', dry_run={args.dry_run})."
    )
    print(f"Log file: {log_path}")

    browser_profile: str = args.profile
    browser_args: list[str] | None = None
    if args.use_chrome_profile:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            raise AutomationError("LOCALAPPDATA is not set; cannot locate Chrome User Data.")
        chrome_user_data = Path(local_app_data) / "Google" / "Chrome" / "User Data"
        if not chrome_user_data.exists():
            raise AutomationError(f"Chrome User Data directory not found: {chrome_user_data}")
        browser_profile = str(chrome_user_data)
        browser_args = [f"--profile-directory={args.use_chrome_profile}"]
        print(
            f"Using system Chrome profile '{args.use_chrome_profile}' at {chrome_user_data}. "
            "(Close existing Chrome windows first if Chrome fails to attach.)"
        )

    imported_cookies: list[dict[str, Any]] = (
        load_cookies_file(args.cookies_file) if args.cookies_file else []
    )

    @browser(
        profile=browser_profile,
        add_arguments=browser_args,
        headless=False,
        remove_default_browser_check_argument=True,
        close_on_crash=False,
        output=None,
        create_error_logs=False,
    )
    def process_chats(driver: Driver, batch: dict[str, Any]) -> dict[str, int]:
        chat_items: list[dict[str, Any]] = batch["items"]
        driver.enable_human_mode()
        stats = {"no_error": 0, "edited": 0, "ambiguous": 0, "failed": 0}
        just_deleted = False
        for idx, chat in enumerate(chat_items, start=1):
            chat_id = str(chat["chat_id"])
            sop_name = str(chat.get("sop_name", chat.get("sop_id", "")))
            print(f"\n[{idx}/{len(chat_items)}] Reviewing {sop_name} (ID {chat_id})...")
            prompt_text = build_evaluation_prompt(chat)

            try:
                ensure_gemini_ready(
                    driver,
                    args.gemini_url,
                    args.model,
                    args.thinking,
                    confirm_model=(idx == 1 and not args.no_model_prompt),
                    is_first_chat=(idx == 1),
                    already_on_new_chat=just_deleted,
                    cookies=imported_cookies if idx == 1 else (),
                )
                just_deleted = False
                raw_text, code_blocks = submit_and_wait_response(
                    driver, prompt_text, args.timeout
                )
                parsed = parse_gemini_response(raw_text, code_blocks)
            except Exception as exc:  # noqa: BLE001
                just_deleted = False
                stats["failed"] += 1
                print(f"  ERROR: {exc}", file=sys.stderr)
                write_log_entry(
                    log_path,
                    {
                        "reviewer": args.reviewer,
                        "chat_id": chat_id,
                        "sop_name": sop_name,
                        "outcome": "failed",
                        "error": str(exc),
                    },
                )
                continue

            log_record: dict[str, Any] = {
                "reviewer": args.reviewer,
                "chat_id": chat_id,
                "sop_name": sop_name,
                "model": args.model,
                "thinking": args.thinking,
                "outcome": parsed.outcome,
                "summary": parsed.summary,
                "raw_text": parsed.raw_text,
            }

            if parsed.outcome == "no_error":
                stats["no_error"] += 1
                print("  Result: No major SOP violations -> marking no_error.")
                if not args.dry_run:
                    client.save_review(args.reviewer, chat_id, "no_error")
            elif parsed.outcome == "edited" and parsed.messages is not None:
                stats["edited"] += 1
                log_record["updated_turn_count"] = len(parsed.messages)
                print(
                    f"  Result: Violations fixed ({len(parsed.messages)} turns) -> "
                    f"updating chat & marking no_error + edited."
                )
                print(f"  Edit reason: {parsed.summary}")
                if not args.dry_run:
                    client.update_chat(args.reviewer, chat_id, parsed.messages)
                    client.save_review(
                        args.reviewer,
                        chat_id,
                        "no_error",
                        reason=parsed.summary,
                        edited=True,
                    )
            else:
                stats["ambiguous"] += 1
                print(
                    "  Result: Ambiguous response (neither clear 'no violations' nor valid JSON). "
                    "Leaving chat pending."
                )

            write_log_entry(log_path, log_record)
            if (
                not args.keep_gemini_chats
                and parsed.outcome in {"no_error", "edited"}
            ):
                if delete_current_gemini_chat(driver):
                    just_deleted = True
                    print("  Deleted Gemini chat from history.")
                else:
                    print(
                        "  Warning: could not auto-delete Gemini chat from sidebar.",
                        file=sys.stderr,
                    )
            if idx < len(chat_items):
                driver.short_random_sleep()

        return stats

    summary_stats = process_chats({"items": queue})
    print(f"\nDone: {summary_stats}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", required=True, help="Reviewer name (e.g. bahaa)")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--gemini-url", default=DEFAULT_GEMINI_URL)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument(
        "--use-chrome-profile",
        nargs="?",
        const="Default",
        default=None,
        help="Use your real Chrome User Data profile (e.g. Default or 'Profile 1'; close Chrome first)",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        default=None,
        help="Optional JSON cookies file to import into the persistent profile",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Gemini model name to select (e.g. Neon, Hydrogen, Pro; default: Neon)",
    )
    parser.add_argument(
        "--thinking",
        default=DEFAULT_THINKING_EFFORT,
        help="Thinking effort level to select (e.g. Low, Medium, High, or '' to leave unchanged; default: Low)",
    )
    parser.add_argument(
        "--add-count",
        type=int,
        default=None,
        help="Add N new chats from the reviewer's SOP pool before reviewing",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max chats to process")
    parser.add_argument("--chat-id", default=None, help="Process only one chat ID")
    parser.add_argument("--include-error", action="store_true", help="Also process chats marked error")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds to wait per Gemini response")
    parser.add_argument("--no-model-prompt", action="store_true", help="Do not pause if model picker isn't matched")
    parser.add_argument(
        "--keep-gemini-chats",
        action="store_true",
        help="Keep Gemini web conversations in history instead of deleting them after saving",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run Gemini & log without updating server")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_batch(args)
    except AutomationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
