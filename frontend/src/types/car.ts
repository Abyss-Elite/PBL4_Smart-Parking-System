export interface Car {
  licensePlate: string;
  dateBooking: string;
  startTimeBooking: string;
  endTimeBooking: string;    
  spotName: string;
  mode: string;
}

export interface CarBookingPending {
  id: string;
  licensePlate: string;
  description: string;
  dateBooking: string;
  startTimeBooking: string;
  endTimeBooking: string;
  checkInTime: string;
  checkOutTime: string;
  imageInUrl: string;
  imageOutUrl: string;
  isOut: boolean;
  isDeleted: boolean;
  fee: number;
  parkingSpot: {
    id: number;
    name: string;
    status: string;
  }
}

export interface Booking {
  id: string;
  code: string;
  totalAmount: number;
  status: string;
  isPaid: boolean;
  cars: any[];
}