import Image from "next/image";
import { SideBarContent } from "./SideBarContent";
import { SideBarFooter } from "./SideBarFooter";
import { useEffect, useState } from "react";
import { userAPI } from "@/api/user/userAPI";

function SideBar({ className }) {
  const ImageLogo =
    "https://cdn.dribbble.com/userupload/28456975/file/original-923692d84cf3a6905b017e91981ed0af.gif";
  const [dataUser, setDataUser] = useState({});

  useEffect(() => {
    const fetchDataUser = async () => {
      try {
        const res = await userAPI.getUser(localStorage.getItem("userId"));
        const data = res.data;
        setDataUser(data);
      } catch (err) {
        console.log("Not fetch data: ", err);
      }
    };
    fetchDataUser();
  }, []);

  return (
    <div
      className={`${className} flex h-screen flex-col justify-between border-r-1 border-[#E3E8EF] text-sm`}
    >
      <div className="flex flex-col items-center gap-6 pt-6">
        <div className="px-6 py-1">
          <Image src={ImageLogo} alt="Logo" className="mr-4 rounded-lg" width={100} height={50} />
        </div>
        <SideBarContent />
      </div>

      <SideBarFooter data={dataUser} />
    </div>
  );
}

export default SideBar;
