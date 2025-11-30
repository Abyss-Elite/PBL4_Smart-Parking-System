"use client";

import React from "react";
import { HistoryItem } from "@/types/history";

interface Props {
  data: HistoryItem[];
  type: "reserved" | "free";
}

export default function HistoryTable({ data, type }: Props) {
  if (!data || data.length === 0) {
    return (
      <div className="text-center text-gray-500 py-10">
        Không có giao dịch nào.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      <table className="min-w-full text-sm text-left">
        <thead className="bg-gray-100 text-gray-700">
          <tr>
            <th className="px-3 py-3">Mã giao dịch</th>
            <th className="px-3 py-3">Biển số</th>
            <th className="px-3 py-3">Tài khoản</th>
            <th className="px-3 py-3">Thời gian vào</th>
            <th className="px-3 py-3">Thời gian ra</th>
            <th className="px-3 py-3">Tổng thời gian</th>
            <th className="px-3 py-3">Số tiền</th>
            <th className="px-3 py-3">Thời gian thanh toán</th>
            {type === "reserved" && (
              <>
                <th className="px-3 py-3">Mã đặt trước</th>
              </>
            )}
          </tr>
        </thead>

        <tbody>
          {data.map((item, index) => (
            <tr
              key={index}
              className="border-b hover:bg-gray-50 transition"
            >
              <td className="px-3 py-3">{item.transactionId}</td>
              <td className="px-3 py-3">{item.licensePlate}</td>
              <td className="px-3 py-3">{item.accountName}</td>
              <td className="px-3 py-3">{item.checkInTime}</td>
              <td className="px-3 py-3">{item.checkOutTime}</td>
              <td className="px-3 py-3">{item.totalTime}</td>
              <td className="px-3 py-3 font-medium">{item.amount}</td>
              <td className="px-3 py-3">{item.paymentTime}</td>
              {type === "reserved" && (
                <>
                  <td className="px-3 py-3">{item.reservedCode}</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
