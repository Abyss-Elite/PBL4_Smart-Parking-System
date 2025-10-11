"use client";

import { useState } from "react";
import LoginForm from "@/components/auth/LoginForm";
import ResetPasswordForm from "@/components/auth/ResetPasswordForm";

export default function LoginPage() {
  const [showReset, setShowReset] = useState(false);

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#6af98e]">
      <div className="animate-fadeIn flex max-h-[90vh] w-[950px] rounded-2xl bg-white p-12 shadow-xl">
        <div className="flex-1 pr-10">
          <div className="mb-10 flex items-center">
            <img
              src="https://cdn.dribbble.com/userupload/28456975/file/original-923692d84cf3a6905b017e91981ed0af.gif"
              alt="Logo"
              className="mr-4 w-24 rounded-lg"
            />
            <h1 className="text-3xl font-bold text-gray-800 ">
              ABYSS - Smart Parking System
            </h1>
          </div>
          <p className="text-lg leading-relaxed text-gray-700">
            Hệ thống bãi đỗ xe thông minh giúp bạn quản lý việc gửi xe dễ dàng hơn.
            <br />
            ✅ Theo dõi tình trạng bãi đỗ xe theo thời gian thực.
            <br />
            ✅ Đặt chỗ trước để đảm bảo có chỗ đỗ.
            <br />
            ✅ Thanh toán nhanh chóng, an toàn.
            <br />✅ Báo cáo chi tiết cho người quản lý.
          </p>
        </div>

        <div className="flex flex-1 items-center justify-center border-l-2 border-gray-200 pl-10">
          {!showReset ? (
            <LoginForm onShowReset={() => setShowReset(true)} />
          ) : (
            <ResetPasswordForm onBack={() => setShowReset(false)} />
          )}
        </div>
      </div>
    </div>
  );
}
