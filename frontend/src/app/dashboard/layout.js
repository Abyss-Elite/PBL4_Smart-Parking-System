"use client";

import SideBar from "@/components/layout/SideBar";
import PATH from "@/routes/PATH";
import { getAccessToken } from "@/utils/tokenStorage";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import BackButton from "@/components/common/BackButton";

function DashBoardLayout({ children }) {
  const router = useRouter();

  useEffect(() => {
    const token = getAccessToken();
    // const token = "123456";
    if (!token) router.replace(PATH.LOGIN);
  }, []);

  return (
    <div className="flex h-screen w-full">
      <SideBar className="w-56 max-w-[14rem] min-w-[14rem] flex-shrink-0" />
      <div className="flex flex-1 overflow-hidden">
        <div className="h-full w-full space-y-4 overflow-auto p-2">
          <BackButton className="mb-2 cursor-pointer" />
          <div>{children}</div>
        </div>
      </div>
    </div>
  );
}

export default DashBoardLayout;
