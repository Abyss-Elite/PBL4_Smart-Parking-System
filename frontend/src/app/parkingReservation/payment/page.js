"use client";

import { useSearchParams } from "next/navigation";

export default function PaymentPage() {
  const searchParams = useSearchParams();

  const id = searchParams.get("id");
  const total = searchParams.get("total");

  return (
    <div className="mx-auto max-w-xl px-6 py-10 space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">Thanh toán</h1>

      <div className="rounded-2xl border bg-white p-6 shadow-sm space-y-4">
        <div className="text-gray-700 space-y-2">
          <p>
            <strong>Mã hóa đơn:</strong> {id}
          </p>
          <p>
            <strong>Số tiền cần thanh toán:</strong>{" "}
            <span className="text-green-600 font-semibold text-xl">
              {Number(total).toLocaleString("vi-VN")}₫
            </span>
          </p>
        </div>
      </div>

      <button className="w-full rounded-xl bg-green-600 py-3 text-white font-medium hover:bg-green-700 transition shadow">
        Xác nhận thanh toán
      </button>
    </div>
  );
}
