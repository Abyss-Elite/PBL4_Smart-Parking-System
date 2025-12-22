import { useEffect, useState } from "react";
import { transaction_historyAPI } from "@/api/transaction-history/transaction-historyAPI";
import HistoryTable from "./HistoryTable";
import { HistoryItem } from "@/types/history";

export default function FreeHistory() {
  
  const [data, setData] = useState<HistoryItem[] | null>(null); 
  const [searchText, setSearchText] = useState("");
  const fetchData = async (query?: string) => {
    try {
      const res = await transaction_historyAPI.getTransactionHistoryOfFreeParking(query);
      setData(res.data);
    } catch (error) {
      console.error("Lỗi khi fetch history free:", error);
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
          placeholder="Nhập Mã giao dịch, Biển số, Tên tài khoản "
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
      <HistoryTable data={data || []} type="free" />
    </>

  );
}
