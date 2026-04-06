import { app, BrowserWindow, ipcMain, screen } from "electron";
import path from "path";

let win: BrowserWindow | null = null;
const nextDevUrl = process.env.NEXT_DEV_URL ?? "http://127.0.0.1:3000";

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
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => app.quit());
ipcMain.on("quit", () => app.quit());
