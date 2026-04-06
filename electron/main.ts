import { app, BrowserWindow, ipcMain, screen } from "electron";
import path from "path";

let win: BrowserWindow | null = null;

function createWindow() {
  if (process.platform === "darwin") {
    app.dock?.hide();
  }

  const winW = 320;
  const winH = 220;
  const { width: screenW } = screen.getPrimaryDisplay().bounds;

  win = new BrowserWindow({
    width: winW,
    height: winH,
    x: screenW - winW - 10,
    y: 20,
    title: "Counter",
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

  // Load a static HTML renderer with a counter. This keeps the example
  // self-contained (no Next.js required).
  const counterHtmlPath = path.join(
    __dirname,
    "..",
    "electron",
    "counter.html"
  );
  win.loadFile(counterHtmlPath);

  win.on("closed", () => {
    win = null;
  });
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => app.quit());
ipcMain.on("quit", () => app.quit());
