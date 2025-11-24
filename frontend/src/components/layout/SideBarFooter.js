import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import PATH from "@/routes/PATH";
import { IconSetting } from "../ui/icon/IconSetting";
import { useEffect, useRef, useState } from "react";
import { handleLogout } from "@/utils/handleLogout";

export function SideBarFooter({ data }) {
  const { username } = data;
  const userImage = "https://photo.znews.vn/w660/Uploaded/mdf_eioxrd/2021_07_06/2.jpg";
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative flex flex-col gap-6 px-4 pb-8" ref={dropdownRef}>
      <Link
        href={PATH.DASHBOARD.SETTING}
        className={`group flex items-center gap-2 rounded-[6] px-2 py-2 transition-colors duration-300 ${
          pathname === PATH.DASHBOARD.SETTING
            ? "bg-[#F3F4F6] dark:bg-gray-700"
            : "hover:bg-[#F3F4F6] dark:hover:bg-gray-800"
        } `}
      >
        <div
          className={`flex items-center gap-2 transition-all duration-300 ${
            pathname === PATH.DASHBOARD.SETTING
              ? "font-semibold text-green-600"
              : "text-[#747778] group-hover:font-semibold group-hover:text-green-600 dark:group-hover:text-gray-100"
          } `}
        >
          <IconSetting />
          <span>Settings</span>
        </div>
      </Link>

      <div
        className="flex cursor-pointer gap-4 border-t-1 border-[#E3E8EF] pt-6 pr-8 pl-2 dark:border-gray-700 "
        onClick={() => setOpen(!open)}
      >
        <Image src={userImage} className="rounded-full" width={32} height={32} alt="imageUser" />
        <div className="flex flex-col justify-center">
          <p className="text-sm font-medium text-[#1F2937] dark:text-gray-100">{username}</p>
          <p className="text-xs text-[#374151] dark:text-gray-100">{data.email}</p>
        </div>
      </div>

      {open && (
        <div className="absolute bottom-[70px] left-2 z-50 w-64 rounded-lg border bg-white p-4 shadow-lg">
          <div className="mb-4 flex gap-3 border-b-1 border-gray-200 pb-4">
            <Image
              src={userImage}
              className="rounded-full"
              width={40}
              height={40}
              alt="imageUser"
            />
            <div>
              <p className="font-semibold text-gray-800">{username}</p>
              <p className="text-sm text-gray-500">{data.email}</p>
            </div>
          </div>

          <Link
            href={PATH.DASHBOARD.PROFILE}
            className="flex items-center gap-3 py-2 text-gray-700 hover:text-green-600"
          >
            <span className="text-sm">Profile</span>
          </Link>

          <button
            className="flex w-full cursor-pointer items-center gap-3 py-2 text-red-600 hover:font-semibold"
            onClick={handleLogout}
          >
            <span className="text-sm">Log out</span>
          </button>
        </div>
      )}
    </div>
  );
}
