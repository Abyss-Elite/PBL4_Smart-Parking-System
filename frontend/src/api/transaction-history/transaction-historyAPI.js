import api from "@/lib/axiosInstance";

export const transaction_historyAPI = {
  getTransactionHistoryOfReservedParking: (query) => {
    return api.get("/transaction/2", {
      params: { query } 
    });
  },
  getTransactionHistoryOfFreeParking: (query) => {
    return api.get("/transaction/1", {
      params: { query }
    });
  },
};
