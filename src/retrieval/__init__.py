"""لایهٔ retrieval: پاک‌سازی دیتاست، BM25، معیارهای بازیابی، و بنچمارک embedding.

ماژول‌های clean/bm25/metrics فقط به numpy و کتابخانهٔ استاندارد وابسته‌اند (تست‌پذیرِ
آفلاین). وابستگی‌های سنگین (torch، sentence-transformers، FlagEmbedding) فقط داخلِ
توابعِ bench به‌صورت lazy import می‌شوند.
"""
