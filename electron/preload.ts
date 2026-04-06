import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("ipc", {
  quit: () => ipcRenderer.send("quit"),
});
