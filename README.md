# ChordLab

אתר קטן לניגון שיר, הצגת אקורדים על ציר זמן, והכנה לחיבור מילים מסונכרנות.

המטרה: לבחור קובץ שמע, לטעון קובץ אקורדים בפורמט JSON, ולראות בזמן אמת איזה אקורד מתנגן.

## איך מפעילים

מהתיקייה של הפרויקט:

```powershell
.venv\Scripts\python.exe -m http.server 8081
```

ואז פותחים בדפדפן:

```text
http://localhost:8081/index.html
```

אם הפורט תפוס, אפשר להריץ על פורט אחר:

```powershell
.venv\Scripts\python.exe -m http.server 8082
```

## מבנה הפרויקט

```text
index.html                  מבנה האתר
style.css                   עיצוב האתר
app.js                      לוגיקת הנגן, ציר האקורדים והמילים

media/
  hero-video.mp4            וידאו הרקע בפריים הראשון
  audio/
    song.mp3                קובץ שמע לדוגמה
    song2.mp3               קובץ שמע ברירת המחדל

data/
  chords/
    *.json                  קבצי אקורדים מוכנים
  lyrics/
    *.json                  קבצי מילים ויישור מילים

chord_detector_madmom.py    זיהוי אקורדים עם madmom בלבד
chord_detector_pro.py       זיהוי אקורדים היברידי: madmom ואז fallback ל-Basic Pitch
chord_detector_basic.py     זיהוי אקורדים בסיסי עם librosa/chroma
chord_detector_bp.py        זיהוי אקורדים דרך Basic Pitch
align_lyrics_auto.py        יישור שורות מילים באופן אוטומטי/משוער
server.py                   ניסוי ראשוני לשרת Python לניתוח אוטומטי
```

## קבצי JSON

קובץ אקורדים צריך להיראות כך:

```json
[
  { "start": 0.0, "end": 2.45, "chord": "Am" },
  { "start": 2.45, "end": 5.07, "chord": "Dm" }
]
```

קובץ מילים צריך להיראות כך:

```json
[
  { "start": 16.0, "end": 20.6, "text": "מדי פעם," },
  { "start": 20.6, "end": 31.7, "text": "רק מדי פעם זה בסדר" }
]
```

אפשר גם לטעון קבצי JSON ידנית מתוך האתר.

## סקריפטים לזיהוי אקורדים

### madmom בלבד

הסקריפט הכי נקי כרגע. משתמש במודל האקורדים של madmom ולא עושה fallback.

```powershell
.venv\Scripts\python.exe chord_detector_madmom.py "media/audio/song2.mp3" --json "data/chords/song2_chords_madmom_only.json" --min-duration 0.5
```

מתי להשתמש:
- כשרוצים תוצאה פשוטה ויציבה יחסית
- כשמספיקים אקורדים בסיסיים כמו `Am`, `Dm`, `G`, `C`

### detector pro

מנסה קודם madmom, ואם הוא נכשל עובר ל-Basic Pitch.

```powershell
.venv\Scripts\python.exe chord_detector_pro.py "media/audio/song2.mp3" --json "data/chords/song2_chords_pro.json" --engine auto --beats 4 --min-duration 0.5
```

מתי להשתמש:
- כשרוצים לנסות תוצאה עשירה יותר
- כשלא אכפת לקבל אקורדים מורחבים כמו `maj7`, `m7b5`, `sus4`

### detector basic

מבוסס librosa/chroma. מהיר יחסית, אבל פחות מדויק.

```powershell
.venv\Scripts\python.exe chord_detector_basic.py "media/audio/song2.mp3" --json "data/chords/song2_chords_basic.json" --grid beats --beats-per-chord 4
```

### Basic Pitch

מזהה תווים עם Basic Pitch ואז מעריך אקורדים לפי חלונות זמן.

```powershell
.venv\Scripts\python.exe chord_detector_bp.py "media/audio/song2.mp3" --json "data/chords/song2_chords_bp.json" --grid beats --beats-per-chord 4 --vocabulary triads
```

## יישור מילים

הסקריפט `align_lyrics_auto.py` לא מזהה מילים מהשמע. הוא מקבל מילים קיימות ומנסה לחלק אותן לזמנים לפי האנרגיה של השיר.

דוגמה:

```powershell
.venv\Scripts\python.exe align_lyrics_auto.py "media/audio/song.mp3" "data/lyrics/lyrics.json" --json "data/lyrics/lyrics_aligned_auto.json" --first-line-start 16 --search-radius 3.0 --min-line-duration 1.0
```

חשוב לדעת:
- זה יישור משוער בלבד
- אם השורה הראשונה מתחילה בזמן ידוע, כדאי להשתמש ב-`--first-line-start`
- בהמשך כדאי להוסיף forced alignment אמיתי עם Whisper/WhisperX

## מה עובד כרגע באתר

- וידאו פתיחה במסך הראשון
- נגן שמע
- בחירת קובץ שמע מהמחשב
- בחירת קובץ אקורדים JSON
- ציר אקורדים שמתעדכן לפי זמן הנגן
- בחירת קובץ מילים JSON
- הצגת מילים לפי זמן

## מהלכים עתידיים

1. לבנות backend מלא שמקבל קובץ שמע ומחזיר אקורדים בלי להריץ סקריפטים ידנית.
2. להוסיף בחירת detector מתוך האתר: `madmom`, `pro`, `basic`, `Basic Pitch`.
3. להוסיף תור עבודות, כי ניתוח אקורדים יכול לקחת זמן.
4. להוסיף יישור מילים אמיתי עם Whisper/forced alignment.
5. להוסיף תמיכה בקישור YouTube דרך backend, אם מחליטים שזה מתאים מבחינת שימוש וזכויות.
6. להוסיף שמירה/עריכה ידנית של אקורדים ומילים מתוך האתר.

## ברירת מחדל באתר

כרגע האתר נטען עם:

```text
media/audio/song2.mp3
data/chords/song2_chords_madmom_only.json
data/lyrics/lyrics.json
```
