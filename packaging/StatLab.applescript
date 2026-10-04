-- StatLab başlatıcı — çift tıklayınca sunucuyu başlatır ve tarayıcıda açar.
set projectDir to "/Users/mac/StatLab"
set appURL to "http://localhost:8501"

-- Sunucu çalışıyor mu? Değilse başlat.
set running to false
try
	do shell script "curl -s -o /dev/null --max-time 2 " & appURL
	set running to true
end try

if not running then
	do shell script "cd " & quoted form of projectDir & " && nohup ./.venv/bin/streamlit run app.py --server.headless true --server.port 8501 >/tmp/statlab_app.log 2>&1 &"
	-- sunucunun ayağa kalkmasını bekle
	repeat 20 times
		delay 1
		try
			do shell script "curl -s -o /dev/null --max-time 2 " & appURL
			exit repeat
		end try
	end repeat
end if

do shell script "open " & appURL
