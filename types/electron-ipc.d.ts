export {};

declare global {
  interface Window {
    ipc?: {
      quit: () => void;
      onUnderlayScreenshot: (handler: (base64Png: string) => void) => () => void;
    };
  }
}
