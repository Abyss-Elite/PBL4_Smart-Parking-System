"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sun, Moon, Globe } from "lucide-react";
import LanguageSetting from "./LanguageSetting";
import React from "react";

export default function SettingPopup({ onClose }) {
  const [activeTab, setActiveTab] = useState<"main" | "theme" | "language">("main");

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 10 }}
        className="absolute bottom-20 left-6 w-56 rounded-xl bg-white dark:bg-gray-800 shadow-lg p-4 z-50"
      >
        {activeTab === "main" && (
          <div className="flex flex-col gap-3">
            <button
              onClick={() => setActiveTab("theme")}
              className="flex items-center gap-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md p-2 transition"
            >
              <Sun className="w-5 h-5" />
              <span>Chế độ Sáng / Tối</span>
            </button>

            <button
              onClick={() => setActiveTab("language")}
              className="flex items-center gap-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md p-2 transition"
            >
              <Globe className="w-5 h-5" />
              <span>Ngôn ngữ</span>
            </button>
          </div>
        )}
        {activeTab === "language" && <LanguageSetting onBack={() => setActiveTab("main")} />}

        <button
          onClick={onClose}
          className="mt-4 w-full text-center text-sm text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
        >
          Đóng
        </button>
      </motion.div>
    </AnimatePresence>
  );
}
