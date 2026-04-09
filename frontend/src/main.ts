import "./styles.css";

type Speaker = {
  speaker: number;
  label: string;
  subtitle: string;
};

const app = document.querySelector<HTMLDivElement>("#app");

if (!app) {
  throw new Error("Missing app root.");
}

function must<T>(value: T | null, name: string): T {
  if (value === null) {
    throw new Error(`Missing ${name}.`);
  }
  return value;
}

const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL || "https://msingi-ai--sauti-tts-inference-inferenceapi-asgi-app.modal.run"
).replace(/\/$/, "");

const speakers: Speaker[] = [
  {
    speaker: 1,
    label: "Female 1",
    subtitle: "Warm and steady",
  },
  {
    speaker: 4,
    label: "Male",
    subtitle: "Grounded and direct",
  },
  {
    speaker: 6,
    label: "Female 2",
    subtitle: "Light and conversational",
  },
];

const examplePrompts = [
  "Mama alienda shambani kupanda mahindi.",
  "Leo tutazungumza kuhusu afya, elimu, na teknolojia kwa lugha ya Kiswahili.",
  "Karibu MsingiAI. Huu ni mfano wa sauti ya Swahili inayozalishwa kwa kutumia Sauti TTS.",
];

app.innerHTML = `
  <main class="min-h-screen">
    <div class="mx-auto max-w-4xl px-4 py-6 sm:px-6 sm:py-8">
      <header class="flex items-center justify-between">
        <div class="inline-flex w-fit items-center gap-3 rounded-full border border-line bg-white/70 px-4 py-2 text-sm text-ink-soft shadow-[0_16px_40px_rgba(42,30,24,0.06)] backdrop-blur">
          <span class="inline-flex h-2.5 w-2.5 rounded-full bg-brand"></span>
          <span class="font-semibold tracking-[0.22em] text-muted uppercase">MsingiAI</span>
          <span class="h-1 w-1 rounded-full bg-line"></span>
          <span class="font-medium">Sauti TTS</span>
        </div>
      </header>

      <section class="mt-6 space-y-4">
        <div class="space-y-3">
          <h1 class="max-w-2xl font-serif text-3xl leading-tight text-foreground sm:text-4xl">
            Fast Swahili text-to-speech.
          </h1>
          <p class="max-w-2xl text-sm leading-7 text-muted sm:text-base">
            Choose a voice and generate audio with Sauti TTS by MsingiAI.
          </p>
          <div class="flex flex-wrap gap-2">
            ${examplePrompts
              .map(
                (prompt, index) => `
                  <button
                    type="button"
                    class="example-btn rounded-full border border-line bg-white/75 px-3 py-2 text-left text-sm text-ink-soft transition hover:border-brand/35 hover:bg-brand-soft hover:text-brand-strong"
                    data-prompt="${index}"
                  >
                    ${prompt}
                  </button>
                `
              )
              .join("")}
          </div>
        </div>
      </section>

      <section class="mt-6">
        <section class="rounded-[28px] border border-line bg-surface p-5 shadow-[var(--shadow)] backdrop-blur sm:p-6">
          <div class="flex items-center justify-between gap-4">
            <div>
              <p class="text-sm font-semibold uppercase tracking-[0.24em] text-muted-soft">Generate</p>
              <h2 class="mt-1 font-serif text-2xl text-foreground sm:text-[2rem]">One voice. One line. One click.</h2>
            </div>
              <p class="hidden text-sm leading-6 text-muted sm:block">Cold starts can take about 30 seconds.</p>
          </div>

          <form id="tts-form" class="mt-6 space-y-5">
            <div class="grid gap-2">
              <label for="speaker" class="text-sm font-semibold text-ink-soft">Voice</label>
              <select
                id="speaker"
                name="speaker"
                required
                class="w-full rounded-2xl border border-line bg-white/80 px-4 py-3 text-base font-medium text-foreground outline-none transition focus:border-brand/50 focus:ring-4 focus:ring-brand/10"
              ></select>
              <p id="speaker-meta" class="text-sm leading-6 text-muted"></p>
            </div>

            <div class="grid gap-2">
              <div class="flex items-center justify-between gap-4">
                <label for="text" class="text-sm font-semibold text-ink-soft">Text</label>
                <span id="text-count" class="text-sm text-muted-soft">0 / 500</span>
              </div>
              <textarea
                id="text"
                name="text"
                rows="6"
                maxlength="500"
                placeholder="Andika sentensi yako hapa kwa Kiswahili."
                required
                class="min-h-[180px] w-full rounded-[24px] border border-line bg-white/85 px-4 py-4 text-base leading-8 text-foreground outline-none transition placeholder:text-muted-soft focus:border-brand/50 focus:ring-4 focus:ring-brand/10"
              ></textarea>
            </div>

            <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <button
                id="submit"
                type="submit"
                class="inline-flex items-center justify-center gap-3 rounded-full bg-brand px-6 py-3 text-base font-semibold text-white shadow-[0_18px_40px_rgba(201,104,66,0.28)] transition hover:bg-brand-strong disabled:cursor-wait disabled:opacity-70"
              >
                <span id="button-spinner" class="hidden h-4 w-4 rounded-full border-2 border-white/35 border-t-white animate-spin"></span>
                <span id="button-text">Generate Audio</span>
              </button>
              <p class="text-sm leading-6 text-muted sm:max-w-xs">Best with short, natural prompts.</p>
            </div>
          </form>

          <div id="status-shell" class="mt-6 rounded-2xl border border-line bg-white/65 px-4 py-3">
            <p id="status-label" class="text-xs font-semibold uppercase tracking-[0.24em] text-muted-soft">Status</p>
            <p id="status" class="mt-1 text-sm leading-6 text-muted" aria-live="polite"></p>
            <div id="loading-bar" class="mt-3 hidden h-1.5 overflow-hidden rounded-full bg-brand/10">
              <div class="h-full w-2/5 rounded-full bg-brand animate-pulse"></div>
            </div>
          </div>

          <section id="audio-shell" class="mt-6 hidden rounded-[28px] border border-line bg-white/80 p-5 shadow-[0_14px_36px_rgba(42,30,24,0.05)]">
            <div class="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p class="text-sm font-semibold uppercase tracking-[0.24em] text-muted-soft">Result</p>
                <h3 class="font-serif text-2xl text-foreground">Your generated clip</h3>
              </div>
              <a
                id="download-mp3"
                class="inline-flex w-fit items-center justify-center rounded-full bg-foreground px-5 py-2.5 text-sm font-semibold text-white transition hover:opacity-90"
                download="sauti-tts.mp3"
              >
                Download MP3
              </a>
            </div>
            <audio id="player" controls class="mt-4 w-full"></audio>
          </section>
        </section>
      </section>
    </div>
  </main>
`;

const form = must(document.querySelector<HTMLFormElement>("#tts-form"), "form");
const speakerSelect = must(document.querySelector<HTMLSelectElement>("#speaker"), "speaker select");
const speakerMeta = must(
  document.querySelector<HTMLParagraphElement>("#speaker-meta"),
  "speaker meta",
);
const textArea = must(document.querySelector<HTMLTextAreaElement>("#text"), "text area");
const textCount = must(document.querySelector<HTMLSpanElement>("#text-count"), "text count");
const status = must(document.querySelector<HTMLParagraphElement>("#status"), "status");
const statusLabel = must(
  document.querySelector<HTMLParagraphElement>("#status-label"),
  "status label",
);
const statusShell = must(document.querySelector<HTMLDivElement>("#status-shell"), "status shell");
const loadingBar = must(document.querySelector<HTMLDivElement>("#loading-bar"), "loading bar");
const submitButton = must(document.querySelector<HTMLButtonElement>("#submit"), "submit button");
const buttonSpinner = must(
  document.querySelector<HTMLSpanElement>("#button-spinner"),
  "button spinner",
);
const buttonText = must(document.querySelector<HTMLSpanElement>("#button-text"), "button text");
const audioShell = must(document.querySelector<HTMLElement>("#audio-shell"), "audio shell");
const player = must(document.querySelector<HTMLAudioElement>("#player"), "audio player");
const downloadMp3 = must(
  document.querySelector<HTMLAnchorElement>("#download-mp3"),
  "download button",
);

let currentAudioUrl: string | null = null;

function populateSpeakers(): void {
  speakerSelect.innerHTML = speakers
    .map(
      (speaker) => `
        <option value="${speaker.speaker}">
          ${speaker.label}
        </option>
      `
    )
    .join("");
  updateSpeakerMeta();
}

function updateSpeakerMeta(): void {
  const selected = speakers.find((item) => item.speaker === Number(speakerSelect.value));
  if (!selected) {
    speakerMeta.textContent = "";
    return;
  }
  speakerMeta.textContent = selected.subtitle;
}

function updateTextCount(): void {
  textCount.textContent = `${textArea.value.length} / 500`;
}

function setStatus(
  message: string,
  tone: "default" | "loading" | "error" | "success" = "default",
): void {
  status.textContent = message;
  statusLabel.textContent =
    tone === "loading"
      ? "Generating"
      : tone === "error"
        ? "Error"
        : tone === "success"
          ? "Ready"
          : "Status";

  status.className =
    tone === "error"
      ? "mt-1 text-sm leading-6 text-error"
      : tone === "success"
        ? "mt-1 text-sm leading-6 text-success"
        : "mt-1 text-sm leading-6 text-muted";
  loadingBar.classList.toggle("hidden", tone !== "loading");

  statusShell.className =
    tone === "loading"
      ? "mt-6 rounded-2xl border border-brand/20 bg-brand-soft px-4 py-3"
      : tone === "error"
        ? "mt-6 rounded-2xl border border-error/20 bg-white/65 px-4 py-3"
        : tone === "success"
          ? "mt-6 rounded-2xl border border-success/20 bg-white/65 px-4 py-3"
          : "mt-6 rounded-2xl border border-line bg-white/65 px-4 py-3";
}

function setLoadingState(isLoading: boolean): void {
  submitButton.disabled = isLoading;
  buttonSpinner.classList.toggle("hidden", !isLoading);
  buttonText.textContent = isLoading ? "Generating..." : "Generate Audio";
}

function clearAudio(): void {
  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = null;
  }
  player.pause();
  player.removeAttribute("src");
  audioShell.classList.add("hidden");
  downloadMp3.removeAttribute("href");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearAudio();

  const text = textArea.value.trim();
  const speaker = Number(speakerSelect.value);
  if (!text) {
    setStatus("Please enter some Swahili text before generating audio.", "error");
    return;
  }

  setLoadingState(true);
  setStatus(
    "Generating audio. First request after idle can take around 30 seconds.",
    "loading",
  );

  try {
    const response = await fetch(`${apiBaseUrl}/v1/synthesize?format=mp3`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, speaker }),
    });

    if (!response.ok) {
      const errorPayload = (await response.json().catch(() => null)) as
        | { detail?: string }
        | null;
      throw new Error(errorPayload?.detail || "Synthesis failed.");
    }

    const audioBlob = await response.blob();
    currentAudioUrl = URL.createObjectURL(audioBlob);
    player.src = currentAudioUrl;
    audioShell.classList.remove("hidden");
    downloadMp3.href = currentAudioUrl;
    setStatus("Audio ready.", "success");
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected error.";
    setStatus(message, "error");
  } finally {
    setLoadingState(false);
  }
});

downloadMp3.addEventListener("click", () => {
  if (!currentAudioUrl) {
    setStatus("MP3 is not ready yet.", "error");
  }
});

document.querySelectorAll<HTMLButtonElement>(".example-btn").forEach((button) => {
  button.addEventListener("click", () => {
    const promptIndex = Number(button.dataset.prompt);
    const prompt = examplePrompts[promptIndex];
    if (!prompt) {
      return;
    }
    textArea.value = prompt;
    updateTextCount();
    textArea.focus();
  });
});

speakerSelect.addEventListener("change", updateSpeakerMeta);
textArea.addEventListener("input", updateTextCount);

populateSpeakers();
updateTextCount();
setStatus("Select a voice and generate audio.", "default");
