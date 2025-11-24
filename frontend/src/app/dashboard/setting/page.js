"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export default function SettingPage() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [language, setLanguage] = useState("vi");

  const toggleTheme = () => {
    setTheme(resolvedTheme === "light" ? "dark" : "light");
  };

  const handleLanguageChange = (e) => {
    const lang = e.target.value;
    setLanguage(lang);
    localStorage.setItem("lang", lang);
  };

  return (
    <div className="bg-gray-50 p-10 transition-colors duration-300 dark:bg-gray-900 overflow-hidden">
      <h1 className="mb-4 text-3xl font-bold text-gray-900 dark:text-gray-100">Settings</h1>
      <p className="mb-10 text-gray-500 dark:text-gray-400">
        Customize your application experience.
      </p>

      <div className="flex max-w-xl flex-col gap-8">

        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm transition-colors duration-300 dark:border-gray-700 dark:bg-gray-800">
          <h2 className="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">
            Appearance
          </h2>

          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900 dark:text-gray-100">Theme</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Switch between light and dark mode.
              </p>
            </div>

            <button
              onClick={toggleTheme}
              className={`rounded-xl px-4 py-2 font-medium transition-all duration-300 ${
                resolvedTheme === "light"
                  ? "bg-gray-900 text-white hover:bg-gray-700"
                  : "bg-gray-200 text-gray-800 hover:bg-gray-300"
              }`}
            >
              {resolvedTheme === "light" ? "Dark Mode" : "Light Mode"}
            </button>
          </div>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm transition-colors duration-300 dark:border-gray-700 dark:bg-gray-800">
          <h2 className="mb-4 text-lg font-semibold text-gray-900 dark:text-gray-100">Language</h2>

          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-gray-900 dark:text-gray-100">App Language</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Choose your preferred language.
              </p>
            </div>

            <select
              value={language}
              onChange={handleLanguageChange}
              className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-gray-900 transition-colors duration-300 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
            >
              <option value="vi">Vietnamese</option>
              <option value="en">English</option>
              <option value="jp">Japanese</option>
              <option value="kr">Korean</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
