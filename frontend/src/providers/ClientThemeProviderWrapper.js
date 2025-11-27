"use client";
import { useState, useEffect } from "react";
import { ThemeProvider } from "./ThemeProvider";

export default function ClientThemeProviderWrapper({ children }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted ? <ThemeProvider>{children}</ThemeProvider> : children;
}
