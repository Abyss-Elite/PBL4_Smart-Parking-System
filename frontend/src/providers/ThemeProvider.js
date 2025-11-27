// "use client";

// import { ThemeProvider as NextThemesProvider } from "next-themes";

// export function ThemeProvider({ children }) {
//   return (
//     <NextThemesProvider attribute="class" defaultTheme="light" enableSystem={true}>
//       {children}
//     </NextThemesProvider>
//   );
// }

"use client";
import { ThemeProvider as NextThemesProvider } from "next-themes";

export function ThemeProvider({ children }) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="light"
      enableSystem={true}
      enableColorScheme={false}
    >
      {children}
    </NextThemesProvider>
  );
}
