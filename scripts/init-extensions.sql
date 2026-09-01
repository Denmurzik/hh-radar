-- Расширения, которые должны существовать до первой миграции.
-- pg_trgm нужен для похожести строк при нормализации навыков,
-- vector -- для столбца эмбеддингов.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
