import json
import streamlit as st
import os

@st.cache_resource
def load_translations():
    """Загружает все файлы переводов из папки locales."""
    translations = {}
    locales_dir = "locales"
    try:
        for filename in os.listdir(locales_dir):
            if filename.endswith(".json"):
                lang_code = filename.split(".")[0]
                with open(os.path.join(locales_dir, filename), "r", encoding="utf-8") as f:
                    translations[lang_code] = json.load(f)
    except FileNotFoundError:
        st.error(f"Папка '{locales_dir}' не найдена. Пожалуйста, создайте ее и добавьте файлы .json с переводами.")
        return {}
    return translations

def get_localizer(translations):
    """Возвращает функцию-переводчик на основе выбранного языка."""
    def t(key, **kwargs):
        # Безопасно читаем язык из session_state, по умолчанию 'ru'
        lang = st.session_state.get('lang', 'ru')
        text = translations.get(lang, {}).get(key, f"<{key}>")
        if kwargs:
            return text.format(**kwargs)
        return text

    return t

all_translations = load_translations()
t = get_localizer(all_translations)