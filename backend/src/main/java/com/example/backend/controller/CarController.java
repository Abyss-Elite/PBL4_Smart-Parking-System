package com.example.backend.controller;
import com.example.backend.model.Car;
import com.example.backend.service.CarService;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PathVariable;
import com.example.backend.DTO.LicensePlateRequest;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
@RestController
@RequestMapping("/api/car")
// @CrossOrigin(origins = "*")
@CrossOrigin(origins = "http://192.168.1.12:5000")


public class CarController {
    @Autowired
    private CarService carService;

    @GetMapping("/{id}")
    public Car getCar(@PathVariable Long id){
        return carService.getCar(id);
    }
    @GetMapping("/all")
    public List<Car> getAllCars(){
        return carService.getAllCars();
    }
    @GetMapping("/byUser/{userId}")
    public List<Car> getCarsByUserId(@PathVariable Long userId){
      
        return carService.getCarsByUserId(userId);
    }

    @PostMapping("/LicensePlateNumber")
    public Car getLicensePlateNumber(@RequestBody LicensePlateRequest licensePlate){
        String plate = licensePlate.getPlate();
        if(plate == null || plate.isEmpty()) throw new RuntimeException("Fail to get license plate number");
        Car car = carService.getCarsByLicensePlateNumber(plate);
        if(car == null) throw new RuntimeException("Car not found with plate: " + plate);
        return car;
    }
}
