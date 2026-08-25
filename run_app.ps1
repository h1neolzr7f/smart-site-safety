Set-Location -LiteralPath $PSScriptRoot

$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
python -m streamlit run app.py --server.port 8502 --server.address 127.0.0.1 --browser.gatherUsageStats false
