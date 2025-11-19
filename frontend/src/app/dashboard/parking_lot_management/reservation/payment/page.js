"use client";

import React, { useContext, useState } from "react";
import { BookingContext } from "@/context/BookingContext";
import PATH from "@/routes/PATH";
import { useRouter } from "next/navigation";

export default function PaymentPage() {
  const [activeTab, setActiveTab] = (useState < "info") | ("otp" > "info");
  const [otp, setOtp] = useState("");
  const router = useRouter();

  const { clearBooking } = useContext(BookingContext);

  const paymentInfo = {
    bank: "NCB",
    cardNumber: "9704 1985 2619 1432 198",
    cardHolder: "Nguyen Van A",
    issueDate: "07/15",
  };

  const qrCodePlaceholder =
    "https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=FakePaymentQRCode";

  const handleOtpChange = () => {
    setOtp(e.target.value);
  };

  const handleConfirm = () => {
    if (!otp) {
      alert("Vui lòng nhập OTP để xác nhận thanh toán");
      return;
    }
    alert("Thanh toán thành công (giả lập)!");
    clearBooking();
    router.push(PATH.DASHBOARD.PARKING_LOT_MANAGEMENT.RESERVATION.HOME);
  };

  return (
    <div className="container mx-auto flex flex-col items-center gap-6 px-4 py-6">
      <h1 className="text-2xl font-bold">Thanh toán đặt chỗ</h1>

      <div className="mb-4 flex gap-2 border-b">
        <button
          className={`px-4 py-2 font-medium ${activeTab === "info" ? "border-b-2 border-green-600 text-green-600" : "text-gray-600"}`}
          onClick={() => setActiveTab("info")}
        >
          Thông tin chuyển khoản
        </button>
        <button
          className={`px-4 py-2 font-medium ${activeTab === "otp" ? "border-b-2 border-green-600 text-green-600" : "text-gray-600"}`}
          onClick={() => setActiveTab("otp")}
        >
          OTP
        </button>
      </div>

      {activeTab === "info" && (
        <div className="flex w-full max-w-md flex-col items-center gap-3 rounded border p-4 shadow-md">
          <img src={qrCodePlaceholder} alt="QR Code" className="h-40 w-40" />
          <p className="text-gray-600">Quét mã QR để thanh toán</p>

          <div className="mt-4 flex w-full flex-col gap-2">
            <p>
              <span className="font-medium">Ngân hàng:</span> {paymentInfo.bank}
            </p>
            <p>
              <span className="font-medium">Số thẻ:</span> {paymentInfo.cardNumber}
            </p>
            <p>
              <span className="font-medium">Chủ thẻ:</span> {paymentInfo.cardHolder}
            </p>
            <p>
              <span className="font-medium">Ngày phát hành:</span> {paymentInfo.issueDate}
            </p>
          </div>
        </div>
      )}

      {activeTab === "otp" && (
        <div className="flex w-full max-w-md flex-col gap-3 rounded border p-4 shadow-md">
          <input
            type="text"
            placeholder="Nhập OTP"
            value={otp}
            onChange={handleOtpChange}
            className="w-full rounded border px-2 py-1"
          />
          <button
            onClick={handleConfirm}
            className="mt-2 rounded bg-green-600 px-4 py-2 text-white hover:bg-green-700"
          >
            Xác nhận thanh toán
          </button>
        </div>
      )}
    </div>
  );
}
