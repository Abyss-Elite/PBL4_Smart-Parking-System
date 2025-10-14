import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import PATH from "@/routes/PATH";
import { IconSetting } from "../ui/icon/IconSetting";

export function SideBarFooter({ data }) {
  const { name, jobTitle } = data;
  const userImage = "https://photo.znews.vn/w660/Uploaded/mdf_eioxrd/2021_07_06/2.jpg";
  const pathname = usePathname();

  return (
    <div className="flex flex-col gap-6 px-4 pb-8">
      <Link
        href={PATH.DASHBOARD.SETTING}
        className={
          pathname === PATH.DASHBOARD.SETTING
            ? "rounded-[6] bg-[#F3F4F6] px-2 py-2"
            : "rounded-[6] px-2 py-2 hover:bg-[#F3F4F6]"
        }
      >
        <div
          className={
            pathname === PATH.DASHBOARD.SETTING
              ? "flex items-center gap-2 font-semibold text-green-600"
              : "flex items-center gap-2 text-[#747778] transition-all hover:font-semibold hover:text-green-600"
          }
        >
          <IconSetting />
          <span>Settings</span>
        </div>
      </Link>
      <div className="flex gap-4 border-t-1 border-[#E3E8EF] pt-6 pr-8 pl-2">
        <Image src={userImage} className="rounded-full" width={32} height={32} alt="imageUser" />
        <div className="flex flex-col justify-center">
          <p className="text-sm font-medium text-[#1F2937]">{name}</p>
          <p className="text-xs text-[#374151]">{jobTitle}</p>
        </div>
      </div>
    </div>
  );
}
