@echo off
REM Quick Tunnel (trycloudflare.com) tro toi CatVTON API local. URL thay doi moi lan chay.
REM Sao chep dong "https://....trycloudflare.com" in ra va dat vao CATVTON_API_URL tren Render.
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000
