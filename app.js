const audio = document.querySelector("#audio");
const currentChord = document.querySelector("#currentChord");
const currentRange = document.querySelector("#currentRange");
const elapsed = document.querySelector("#elapsed");
const duration = document.querySelector("#duration");
const progressFill = document.querySelector("#progressFill");
const audioFile = document.querySelector("#audioFile");
const chordsFile = document.querySelector("#chordsFile");
const audioFileName = document.querySelector("#audioFileName");
const beatsPerChordInput = document.querySelector("#beatsPerChord");
const timeline = document.querySelector("#timeline");
const chordCount = document.querySelector("#chordCount");
const upNext = document.querySelector("#upNext");
const nextLabel = document.querySelector("#nextLabel");
const lyricsLine = document.querySelector("#lyricsLine");
const lyricsStatus = document.querySelector("#lyricsStatus");

let chords = [];
let lyrics = [];
let activeIndex = -1;
let activeLyricIndex = -1;
let selectedAudioUrl = null;

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) {
    return "0:00";
  }

  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

function findChordIndex(time) {
  if (!chords.length) {
    return -1;
  }

  const exactIndex = chords.findIndex((item) => time >= item.start && time < item.end);
  if (exactIndex !== -1) {
    return exactIndex;
  }

  if (time < chords[0].start) {
    return 0;
  }

  return chords.length - 1;
}

function findChordAt(time) {
  const index = findChordIndex(time);
  return index >= 0 ? chords[index]?.chord : "";
}

function estimateLyricTimes() {
  if (!lyrics.length) {
    return;
  }

  const lastChordEnd = chords.length ? chords[chords.length - 1].end : audio.duration || 0;
  const songStart = chords[0]?.start || 0;
  const usableDuration = Math.max(lastChordEnd - songStart, lyrics.length);
  const lineDuration = usableDuration / lyrics.length;

  lyrics = lyrics.map((line, index) => {
    if (Number.isFinite(line.start) && Number.isFinite(line.end)) {
      return line;
    }

    const start = songStart + index * lineDuration;
    const end = index === lyrics.length - 1 ? lastChordEnd : start + lineDuration;
    return { ...line, start, end, estimated: true };
  });
}

function findLyricIndex(time) {
  if (!lyrics.length) {
    return -1;
  }

  const exactIndex = lyrics.findIndex((line) => time >= line.start && time < line.end);
  if (exactIndex !== -1) {
    return exactIndex;
  }

  if (time < lyrics[0].start) {
    return 0;
  }

  return lyrics.length - 1;
}

function renderLyricLine(index, currentTime) {
  const line = lyrics[index];
  if (!line) {
    lyricsLine.innerHTML = "";
    return;
  }

  const words = line.text.split(/\s+/).filter(Boolean);
  const lineDuration = Math.max(line.end - line.start, words.length * 0.25);

  lyricsLine.innerHTML = words
    .map((word, wordIndex) => {
      const wordStart = line.start + (lineDuration * wordIndex) / Math.max(words.length, 1);
      const chord = findChordAt(wordStart);
      return `<span class="word" data-chord="${chord}">${word}</span>`;
    })
    .join(" ");

  const currentWord = Math.min(
    words.length - 1,
    Math.max(0, Math.floor(((currentTime - line.start) / lineDuration) * words.length)),
  );
  const wordElements = lyricsLine.querySelectorAll(".word");
  wordElements[currentWord]?.classList.add("active-word");

  if (lyricsStatus) {
    const mode = line.estimated ? "תזמון משוער" : "תזמון מדויק";
    lyricsStatus.textContent = `${mode}: שורה ${index + 1} מתוך ${lyrics.length}`;
  }

  activeLyricIndex = index;
}

function renderTimeline() {
  const total = chords.length ? chords[chords.length - 1].end : 1;
  timeline.innerHTML = "";

  chords.forEach((item, index) => {
    const segment = document.createElement("button");
    const width = Math.max(((item.end - item.start) / total) * 2600, 92);
    segment.className = "segment";
    segment.type = "button";
    segment.style.setProperty("--segment-width", `${width}px`);
    segment.dataset.index = index;
    segment.innerHTML = `
      <strong>${item.chord}</strong>
      <span>${formatTime(item.start)}-${formatTime(item.end)}</span>
    `;
    segment.addEventListener("click", () => {
      audio.currentTime = item.start;
      audio.play();
    });
    timeline.appendChild(segment);
  });

  chordCount.textContent = `${chords.length} מקטעי אקורדים`;
}

function setChords(nextChords) {
  chords = nextChords;
  activeIndex = -1;
  renderTimeline();
  setActiveChord(findChordIndex(audio.currentTime));
  estimateLyricTimes();
  renderLyricLine(findLyricIndex(audio.currentTime), audio.currentTime);
}

function renderUpNext(index) {
  const nextChords = chords.slice(index + 1, index + 5);
  upNext.innerHTML = "";

  if (!nextChords.length) {
    nextLabel.textContent = "אין עוד אקורדים אחרי הנקודה הזו.";
    return;
  }

  nextLabel.textContent = "האקורדים הקרובים:";
  nextChords.forEach((item) => {
    const card = document.createElement("div");
    card.className = "next-card";
    card.innerHTML = `
      <strong>${item.chord}</strong>
      <span>${formatTime(item.start)}-${formatTime(item.end)}</span>
    `;
    upNext.appendChild(card);
  });
}

function updateLyricsChord(chord) {
  const activeWord = lyricsLine.querySelector(".active-word");
  if (activeWord) {
    activeWord.dataset.chord = chord;
  }
}

function setActiveChord(index) {
  if (index === activeIndex || index < 0) {
    return;
  }

  const previous = timeline.querySelector(".segment.active");
  if (previous) {
    previous.classList.remove("active");
  }

  const segment = timeline.querySelector(`[data-index="${index}"]`);
  if (segment) {
    segment.classList.add("active");
    segment.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
  }

  const item = chords[index];
  currentChord.textContent = item.chord;
  currentRange.textContent = `${formatTime(item.start)} עד ${formatTime(item.end)}`;
  updateLyricsChord(item.chord);
  renderUpNext(index);
  activeIndex = index;
}

function updatePlaybackUi() {
  const currentTime = audio.currentTime;
  const audioDuration = audio.duration || (chords.length ? chords[chords.length - 1].end : 0);
  const index = findChordIndex(currentTime);
  const lyricIndex = findLyricIndex(currentTime);

  elapsed.textContent = formatTime(currentTime);
  duration.textContent = formatTime(audioDuration);
  progressFill.style.width = audioDuration ? `${(currentTime / audioDuration) * 100}%` : "0%";
  setActiveChord(index);
  renderLyricLine(lyricIndex, currentTime);
}

function loadSelectedAudioPreview(file) {
  if (!file) {
    return;
  }

  if (selectedAudioUrl) {
    URL.revokeObjectURL(selectedAudioUrl);
  }

  selectedAudioUrl = URL.createObjectURL(file);
  audio.pause();
  audio.src = selectedAudioUrl;
  audio.load();
  activeIndex = -1;
  activeLyricIndex = -1;

  elapsed.textContent = "0:00";
  duration.textContent = "0:00";
  progressFill.style.width = "0%";
  currentRange.textContent = "קובץ חדש נטען. האקורדים עדיין מגיעים מ-chords.json";

  if (audioFileName) {
    audioFileName.textContent = `נטען כרגע: ${file.name}`;
  }

  updatePlaybackUi();
}

async function analyzeSelectedAudio(file) {
  if (!file) {
    return;
  }

  const formData = new FormData();
  formData.append("audio", file);
  formData.append("beatsPerChord", beatsPerChordInput?.value || "4");

  if (audioFileName) {
    audioFileName.textContent = `מנתח אקורדים עבור: ${file.name}`;
  }
  currentRange.textContent = "מעלה קובץ ומחשב אקורדים...";

  try {
    const response = await fetch("/api/analyze-audio", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }

    if (selectedAudioUrl) {
      URL.revokeObjectURL(selectedAudioUrl);
      selectedAudioUrl = null;
    }

    audio.pause();
    audio.src = result.audioUrl;
    audio.load();
    setChords(result.chords);
    elapsed.textContent = "0:00";
    duration.textContent = "0:00";
    progressFill.style.width = "0%";

    if (audioFileName) {
      audioFileName.textContent = `נטען ונותח: ${file.name} (${Math.round(result.tempo)} BPM)`;
    }
    currentRange.textContent = `נוצרו ${result.chords.length} מקטעי אקורדים`;
  } catch (error) {
    loadSelectedAudioPreview(file);
    if (audioFileName) {
      audioFileName.textContent = `נטען מקומית ללא אקורדים חדשים: ${file.name}`;
    }
    currentRange.textContent = `השרת לא ניתח את הקובץ: ${error.message}`;
  }
}

function validateChordRows(rows) {
  return Array.isArray(rows) && rows.every((row) => (
    Number.isFinite(row.start)
    && Number.isFinite(row.end)
    && typeof row.chord === "string"
  ));
}

async function loadSelectedChordsFile(file) {
  if (!file) {
    return;
  }

  try {
    const text = await file.text();
    const parsed = JSON.parse(text);

    if (!validateChordRows(parsed)) {
      throw new Error("הקובץ צריך להיות מערך של start/end/chord");
    }

    setChords(parsed);
    currentRange.textContent = `נטענו ${parsed.length} אקורדים מתוך ${file.name}`;
    chordCount.textContent = `${parsed.length} מקטעי אקורדים (${file.name})`;
  } catch (error) {
    currentRange.textContent = `לא הצלחתי לטעון JSON: ${error.message}`;
  }
}

async function loadChords() {
  try {
    const response = await fetch(`chords.json?v=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    chords = await response.json();
    renderTimeline();
    setActiveChord(0);
  } catch (error) {
    currentChord.textContent = "!";
    currentRange.textContent = "לא הצלחתי לטעון chords.json";
    chordCount.textContent = error.message;
  }
}

async function loadLyrics() {
  try {
    const response = await fetch(`lyrics.json?v=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    lyrics = await response.json();
    estimateLyricTimes();
    renderLyricLine(0, 0);
  } catch (error) {
    if (lyricsStatus) {
      lyricsStatus.textContent = `לא הצלחתי לטעון lyrics.json: ${error.message}`;
    }
  }
}

audio.addEventListener("loadedmetadata", updatePlaybackUi);
audio.addEventListener("timeupdate", updatePlaybackUi);
audio.addEventListener("play", updatePlaybackUi);
audio.addEventListener("seeked", updatePlaybackUi);
audioFile?.addEventListener("change", (event) => {
  analyzeSelectedAudio(event.target.files?.[0]);
});
chordsFile?.addEventListener("change", (event) => {
  loadSelectedChordsFile(event.target.files?.[0]);
});

async function init() {
  await loadChords();
  await loadLyrics();
  estimateLyricTimes();
  updatePlaybackUi();
}

init();
