"use client";
import { useState } from "react";
import Image from "next/image";
import { IconSetting } from "../ui/icon/IconSetting";
import SettingPopup from "./SettingPopup";

export function SideBarFooter({ data }) {
  const { name, jobTitle } = data;
  const userImage = "https://photo.znews.vn/w660/Uploaded/mdf_eioxrd/2021_07_06/2.jpg";
  const [showSetting, setShowSetting] = useState(false);

  return (
    <div className="relative flex flex-col gap-6 px-4 pb-8">
      <button
        onClick={() => setShowSetting((prev) => !prev)}
        className="rounded-[6] px-2 py-2 hover:bg-[#F3F4F6] transition-all"
      >
        <div className="flex items-center gap-2 text-[#747778] hover:font-semibold hover:text-green-600">
          <IconSetting />
          <span>Settings</span>
        </div>
      </button>

      {showSetting && (
        <SettingPopup onClose={() => setShowSetting(false)} />
      )}

      <div className="flex gap-4 border-t border-[#E3E8EF] pt-6 pr-8 pl-2">
        <Image src={userImage} className="rounded-full" width={32} height={32} alt="imageUser" />
        <div className="flex flex-col justify-center">
          <p className="text-sm font-medium text-[#1F2937]">{name}</p>
          <p className="text-xs text-[#374151]">{jobTitle}</p>
        </div>
      </div>
    </div>
  );
}
