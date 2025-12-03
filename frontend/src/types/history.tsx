export type ParkingType = "reserved" | "free";

export interface HistoryItem {
  transactionId: string;
  amount: number;
  accountName: string;
  paymentTime: string;  
  licensePlate?: string;
  checkInTime?: string;
  checkOutTime?: string;
  reservedTime?: string; 
  totalTime?: number;
  reservedCode?: string;    
}
