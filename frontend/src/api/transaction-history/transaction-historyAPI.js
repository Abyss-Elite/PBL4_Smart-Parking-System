export const transaction_historyAPI = {
  getTransactionHistoryOfReservedParking: (query) => {
    return api.get("/transactionHistory/1", {
      params: { query } 
    });
  },
  getTransactionHistoryOfFreeParking: (query) => {
    return api.get("/transactionHistory/2", {
      params: { query }
    });
  },
};
