package com.example.backend.service;
import com.example.backend.model.Car;
import com.example.backend.repository.CarRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.Optional;
@Service

public class CarService {
    @Autowired
    private CarRepository carRepository;
    
    public List<Car> getAllCars(){
        return carRepository.findAll();
    }

    public Car getCar(Long id){
        Optional<Car> carOp  = carRepository.findById(id);
        Car car = carOp.orElse(null);
        return car;
    }
    
    public Car createCar(Car car){
        return carRepository.save(car);
    }

    public Car updateCar(Long id, Car carDetails) {
        Car car = carRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Car not found"));

        if (carDetails.getLicensePlate() != null && !carDetails.getLicensePlate().isBlank()) {
            car.setLicensePlate(carDetails.getLicensePlate());
        }

        if (carDetails.getDescription() != null) {
            car.setDescription(carDetails.getDescription());
        }

        if (carDetails.getIsOut() != null) {
            car.setIsOut(carDetails.getIsOut());
        }

        if (carDetails.getImageUrl() != null) {
            car.setImageUrl(carDetails.getImageUrl());
        }

        if (carDetails.getStatus() != null && !carDetails.getStatus().isBlank()) {
            car.setStatus(carDetails.getStatus());
        }

        if (carDetails.getRegistrationDate() != null) {
            car.setRegistrationDate(carDetails.getRegistrationDate());
        }

        if (carDetails.getIsDelete() != null) {
            car.setIsDelete(carDetails.getIsDelete());
        }

        if (carDetails.getLastTime() != null) {
            car.setLastTime(carDetails.getLastTime());
        }

        return carRepository.save(car);
    }

    public void deleteCar(Long id) {
        Car car = carRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Xe không tồn tại, không thể xóa"));

        carRepository.delete(car);
    }

    public List<Car> getCarsByUserId(Long userId){
        return carRepository.findByOwner_Id(userId);
    }
    public Car getCarsByLicensePlateNumber(String plate){
        // http://192.168.1.124/capture
        // Car car = carRepository.findByLicensePlate(plate);
        // String captureUrl = ""
        // if(car == null) 
        return carRepository.findByLicensePlate(plate);
    }
}
