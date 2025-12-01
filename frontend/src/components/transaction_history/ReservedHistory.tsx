import { useEffect, useState } from "react";
import { transaction_historyAPI } from "@/api/transaction-history/transaction-historyAPI";
import HistoryTable from "./HistoryTable";
import { HistoryItem } from "@/types/history";

export default function ReservedHistory() {
  const mockReservedHistory: HistoryItem[] = [
    {
      licensePlate: "30A-123.45",
      checkInTime: "2025-11-29 08:30",
      checkOutTime: "2025-11-29 10:45",
      paymentTime: "2025-11-29 08:35",
      totalTime: 135, 
      amount: 45000,
      transactionId: "TXN001",
      accountName: "Nguyễn Văn A",
      reservedCode: "RES12345",
    },
    {
      licensePlate: "79B-987.65",
      checkInTime: "2025-11-29 09:15",
      checkOutTime: "2025-11-29 11:00",
      paymentTime: "2025-11-29 08:35",
      totalTime: 105,
      amount: 35000,
      transactionId: "TXN002",
      accountName: "Trần Thị B",
      reservedCode: "RES67890",
    },
    {
      licensePlate: "43C-246.80",
      checkInTime: "2025-11-29 10:00",
      checkOutTime: "2025-11-29 12:30",
      paymentTime: "2025-11-29 08:35",
      totalTime: 150,
      amount: 50000,
      transactionId: "TXN003",
      accountName: "Lê Văn C",
      reservedCode: "RES13579",
    },
  ];
  const [data, setData] = useState<HistoryItem[] | null>(null); 
  const [searchText, setSearchText] = useState("");
  const fetchData = async (query?:string) => {
    try {
      const res = await transaction_historyAPI.getTransactionHistoryOfReservedParking(query);
      setData(res.data);
      // setData(mockReservedHistory);
    } catch (error) {
      console.error("Lỗi khi fetch history reserved:", error);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleSearch = () =>{
    fetchData(searchText);
  }

  return (
    <>
      <div className="flex items-center gap-2 mb-4">
        <input
          type="text"
          value={searchText}
          placeholder="Nhập Mã giao dịch, Biển số, Tên tài khoản hoặc Mã đặt trước"
          onChange={(e) => setSearchText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSearch();
          }}
          className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
        <button
          onClick={handleSearch}
          className="px-5 py-2 bg-blue-500 text-white font-medium rounded-md hover:bg-blue-600 transition-colors"
        >
          Tìm kiếm
        </button>
      </div>
      <HistoryTable data={data || []} type="reserved" />;
    </>

  );
}
