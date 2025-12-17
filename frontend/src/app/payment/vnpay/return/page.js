"use client";

import { useEffect, useState, startTransition } from "react";
import { CheckCircle2, CreditCard, Landmark, Calendar, FileText } from "lucide-react";
import { formatCurrency } from "@/utils/formatCurrency";
import PATH from "@/routes/PATH";

export default function ReturnPage() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const obj = {};
    params.forEach((v, k) => (obj[k] = v));

    startTransition(() => setData(obj));
  }, []);

  if (!data) return null;

  const amount = formatCurrency(Number(data.vnp_Amount) / 100);
  const formatDate = (date) => {
    if (!date) return "";
    const y = date.substring(0, 4);
    const m = date.substring(4, 6);
    const d = date.substring(6, 8);
    const hh = date.substring(8, 10);
    const mm = date.substring(10, 12);
    const ss = date.substring(12, 14);
    return `${d}/${m}/${y} ${hh}:${mm}:${ss}`;
  };

  const items = [
    {
      label: "Số tiền",
      value: amount + " VNĐ",
      icon: <CreditCard className="h-5 w-5 text-blue-600" />,
    },
    {
      label: "Ngân hàng",
      value: data.vnp_BankCode,
      icon: <Landmark className="h-5 w-5 text-purple-600" />,
    },
    {
      label: "Mã giao dịch ngân hàng",
      value: data.vnp_BankTranNo,
      icon: <CreditCard className="h-5 w-5 text-green-600" />,
    },
    {
      label: "Nội dung đơn hàng",
      value: data.vnp_OrderInfo,
      icon: <FileText className="h-5 w-5 text-orange-600" />,
    },
    {
      label: "Thời gian thanh toán",
      value: formatDate(data.vnp_PayDate),
      icon: <Calendar className="h-5 w-5 text-red-600" />,
    },
  ];

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-100 p-4">
      <div className="w-full max-w-md rounded-xl bg-white p-6 shadow-lg">
        <div className="mb-5 flex flex-col items-center">
          <CheckCircle2 className="h-14 w-14 text-green-500" />
          <h2 className="mt-3 text-xl font-bold text-gray-800">Thanh toán thành công</h2>
          <p className="mt-1 text-center text-sm text-gray-500">Cảm ơn bạn đã sử dụng dịch vụ</p>
        </div>

        <div className="space-y-3">
          {items.map((item, idx) => (
            <div
              key={idx}
              className="flex items-center gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3"
            >
              {item.icon}
              <div>
                <p className="text-sm text-gray-500">{item.label}</p>
                <p className="text-[15px] font-semibold text-gray-800">{item.value}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6">
          <a
            href="http://localhost:3000/parkingReservation"
            className="block rounded-lg bg-blue-600 py-2 text-center font-medium text-white transition hover:bg-blue-700"
          >
            Trở về trang đặt chỗ
          </a>
        </div>

        <p className="mt-4 text-center text-xs text-gray-500">
          Backend cần xác minh chữ ký tại{" "}
          <code className="rounded bg-gray-200 px-1">/api/v1/vnpay/return</code>
        </p>
      </div>
    </div>
  );
}
