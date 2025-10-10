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
    public Car getCar(Long id){
        Optional<Car> carOp  = carRepository.findById(id);
        Car car = carOp.orElse(null);
        return car;
    }
    public List<Car> getAllCars(){
        return carRepository.findAll();
    }
    public List<Car> getCarsByUserId(Long userId){
        return carRepository.findByOwner_Id(userId);
    }
    public Car getCarsByLicensePlateNumber(String plate){
        return carRepository.findByLicensePlate(plate);
    }
}
