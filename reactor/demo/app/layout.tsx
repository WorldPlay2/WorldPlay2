import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "WorldPlay2 on Reactor",
  description: "Explore a generated world from an image and a prompt with WASD and the arrow keys.",
  icons: { icon: "/logo/symbol-white.svg" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased font-sans">{children}</body>
    </html>
  );
}
