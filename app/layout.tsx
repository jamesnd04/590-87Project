import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Game Overlay",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full bg-transparent">{children}</body>
    </html>
  );
}
