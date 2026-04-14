import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("ipc", {
  quit: () => ipcRenderer.send("quit"),
  onUnderlayScreenshot: (handler: (base64Png: string) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, data: string) => {
      handler(data);
    };
    ipcRenderer.on("underlay-screenshot", listener);
    return () => {
      ipcRenderer.removeListener("underlay-screenshot", listener);
    };
  },
});
