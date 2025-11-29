export type ParkingType = "reserved" | "free";

export interface HistoryItem {
  licensePlate: string;
  checkInTime: string;
  checkOutTime?: string;
  reservedTime?: string; 
  paymentTime: string;  
  totalTime: number;
  amount: number;
  transactionId: string;
  accountName: string;
  reservedCode?: string;    
}
