"use client";
import { ArrowLeft } from "lucide-react";
import React from "react";
import { useState } from "react";

export default function LanguageSetting({ onBack }) {
  const [lang, setLang] = useState("vi");

  return (
    <div className="flex flex-col gap-3">
      <button onClick={onBack} className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700">
        <ArrowLeft className="w-4 h-4" /> Quay lại
      </button>

      <p className="font-medium mt-2">Chọn ngôn ngữ hiển thị:</p>

      <div className="flex flex-col gap-2">
        <label className="flex items-center gap-2">
          <input
            type="radio"
            checked={lang === "vi"}
            onChange={() => setLang("vi")}
          />
          Tiếng Việt
        </label>
        <label className="flex items-center gap-2">
          <input
            type="radio"
            checked={lang === "en"}
            onChange={() => setLang("en")}
          />
          English
        </label>
      </div>
    </div>
  );
}
