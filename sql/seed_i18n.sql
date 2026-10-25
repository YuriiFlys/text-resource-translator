-- Тестові дані для локальної розробки: 3 локалі, 10 рядків інтерфейсу,
-- англійська повна, українська часткова, польська порожня.
INSERT INTO i18n_locale (locale_code, display_name, enabled) VALUES
    ('en', 'English', true),
    ('uk', 'Українська', true),
    ('pl', 'Polski', true),
    ('de', 'Deutsch', false)
ON CONFLICT (locale_code) DO NOTHING;

INSERT INTO i18n_text (text_code, owner_kind) VALUES
    ('common.save',                 'button'),
    ('common.cancel',               'button'),
    ('common.delete',               'button'),
    ('settings.profile.title',      'heading'),
    ('settings.profile.save_hint',  'hint'),
    ('settings.profile.delete_confirm', 'dialog'),
    ('auth.login.username',         'label'),
    ('auth.login.password',         'label'),
    ('auth.login.error_invalid',    'error'),
    ('files.upload.progress',       'status')
ON CONFLICT (text_code) DO NOTHING;

INSERT INTO i18n_translation (text_id, locale_code, translated_text)
SELECT t.text_id, v.locale_code, v.translated_text
FROM (VALUES
    ('common.save',                 'en', 'Save'),
    ('common.cancel',               'en', 'Cancel'),
    ('common.delete',               'en', 'Delete'),
    ('settings.profile.title',      'en', 'Profile settings'),
    ('settings.profile.save_hint',  'en', 'Changes are saved automatically every {0} seconds.'),
    ('settings.profile.delete_confirm', 'en', 'Are you sure you want to delete your profile? This action cannot be undone.'),
    ('auth.login.username',         'en', 'Username'),
    ('auth.login.password',         'en', 'Password'),
    ('auth.login.error_invalid',    'en', 'Invalid username or password.'),
    ('files.upload.progress',       'en', 'Uploading {{fileName}}: {{percent}}%'),
    ('common.save',                 'uk', 'Зберегти'),
    ('common.cancel',               'uk', 'Скасувати'),
    ('settings.profile.title',      'uk', 'Налаштування профілю'),
    ('auth.login.username',         'uk', 'Ім''я користувача'),
    ('auth.login.password',         'uk', 'Пароль')
) AS v(text_code, locale_code, translated_text)
JOIN i18n_text t ON t.text_code = v.text_code
ON CONFLICT (text_id, locale_code) DO NOTHING;
