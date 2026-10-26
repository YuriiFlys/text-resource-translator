"""Фейкова chat-модель для тестів: перекладає за словником, може зіпсувати відповідь раз."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class DictionaryFakeChat(BaseChatModel):
    dictionary: dict[str, str]
    break_once: set[str] = set()  # text_code, для яких перша відповідь зіпсує плейсхолдер
    calls: int = 0
    fence: bool = False  # загорнути відповідь у ```json ... ```

    @property
    def _llm_type(self) -> str:
        return "dictionary-fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any):
        self.calls += 1
        payload = json.loads(messages[-1].content)
        items = {}
        for s in payload["strings_to_translate"]:
            translation = self.dictionary.get(
                s["source"], f"[{payload['target_locale']}] {s['source']}"
            )
            if s["text_code"] in self.break_once:
                translation = translation.replace("{", "").replace("}", "")
                self.break_once = self.break_once - {s["text_code"]}
            items[s["id"]] = translation
        text = json.dumps({"items": items}, ensure_ascii=False)
        if self.fence:
            text = f"```json\n{text}\n```"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
