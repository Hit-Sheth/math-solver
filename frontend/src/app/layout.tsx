import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Handwritten Math Solver",
  description: "Draw digits and operators and solve them using AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
