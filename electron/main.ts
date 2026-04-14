import {
  app,
  BrowserWindow,
  desktopCapturer,
  globalShortcut,
  ipcMain,
  screen,
} from "electron";
import path from "path";

let win: BrowserWindow | null = null;
let capturingUnderlay = false;
const nextDevUrl = process.env.NEXT_DEV_URL ?? "http://127.0.0.1:3000";

/** Global shortcut: Cmd+Shift+G (mac) / Ctrl+Shift+G (Win/Linux). */
const CAPTURE_UNDERLAY_ACCELERATOR = "CommandOrControl+Shift+G";

async function capturePrimaryDisplayPng(): Promise<Buffer> {
  const primary = screen.getPrimaryDisplay();
  const { width, height } = primary.bounds;
  const scale = primary.scaleFactor;
  const sources = await desktopCapturer.getSources({
    types: ["screen"],
    thumbnailSize: {
      width: Math.max(1, Math.floor(width * scale)),
      height: Math.max(1, Math.floor(height * scale)),
    },
  });
  if (sources.length === 0) {
    throw new Error("No screen sources available for capture.");
  }
  const idStr = String(primary.id);
  const source =
    sources.find((s) => String(s.display_id) === idStr) ?? sources[0];
  return source.thumbnail.toPNG();
}

async function captureUnderlayAndSend(): Promise<void> {
  if (capturingUnderlay || !win || win.isDestroyed()) return;
  capturingUnderlay = true;
  try {
    win.hide();
    await new Promise<void>((resolve) => setTimeout(resolve, 280));
    const png = await capturePrimaryDisplayPng();
    const b64 = png.toString("base64");
    if (win && !win.isDestroyed()) {
      win.webContents.send("underlay-screenshot", b64);
    }
  } catch (e) {
    console.error("[overlay] underlay capture failed:", e);
  } finally {
    capturingUnderlay = false;
    if (win && !win.isDestroyed()) {
      win.show();
    }
  }
}

function registerCaptureShortcut(): void {
  const ok = globalShortcut.register(CAPTURE_UNDERLAY_ACCELERATOR, () => {
    void captureUnderlayAndSend();
  });
  if (!ok) {
    console.error(
      `[overlay] Failed to register hotkey ${CAPTURE_UNDERLAY_ACCELERATOR}`,
    );
  }
}

function createWindow() {
  if (process.platform === "darwin") {
    app.dock?.hide();
  }

  const winW = 320;
  const { width: screenW, height: screenH } = screen.getPrimaryDisplay().bounds;
  const winH = screenH - 36;

  win = new BrowserWindow({
    width: winW,
    height: winH,
    x: screenW - winW,
    y: 0,
    title: "Overlay",
    transparent: true,
    backgroundColor: "#00000000",
    frame: false,
    alwaysOnTop: true,
    fullscreenable: false,
    resizable: false,
    skipTaskbar: true,
    hasShadow: false,
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, "preload.js"),
    },
  });

  // Ensure the overlay follows Space/Desktop switching (and full-screen).
  win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  // Raise overlay above "normal" windows.
  win.setAlwaysOnTop(true, "screen-saver");

  win.loadURL(nextDevUrl);

  win.on("closed", () => {
    win = null;
  });

  registerCaptureShortcut();
}

app.whenReady().then(createWindow);
app.on("will-quit", () => {
  globalShortcut.unregisterAll();
});
app.on("window-all-closed", () => app.quit());
ipcMain.on("quit", () => app.quit());
