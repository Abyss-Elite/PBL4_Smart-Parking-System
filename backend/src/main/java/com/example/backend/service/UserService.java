package com.example.backend.service;
import com.example.backend.model.User;
import com.example.backend.model.Car;
import com.example.backend.model.Role;
import com.example.backend.repository.CarRepository;
import com.example.backend.repository.UserRepository;
import com.example.backend.validator.UserValidator;
import com.example.backend.repository.RoleRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Optional;
import java.util.List;
@Service
public class UserService {
    @Autowired
    private CarRepository carRepository;
    @Autowired
    private RoleRepository roleRepository;
    private final UserRepository userRepository;
    private final UserValidator userValidator;
    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    public UserService(UserRepository userRepository, RoleRepository roleRepository, UserValidator userValidator){
        this.userRepository = userRepository;
        this.roleRepository = roleRepository;
        this.userValidator = userValidator;
    }

    public User register(String username, String password, Role role, String email, String phoneNumber, String status, Boolean isDelete){
        userValidator.validateEmail(email, null);
        userValidator.validatePassword(password);
        userValidator.validatePhoneNumber(phoneNumber, null);
        userValidator.validateUsername(username, null);
        User user = new User();
        user.setUsername(username);
        user.setPassword(passwordEncoder.encode(password));
        user.setRole(role);
        user.setEmail(email);
        user.setStatus(status);
        user.setDelete(isDelete);
        user.setPhoneNumber(phoneNumber);

        return userRepository.save(user);
    }

    public boolean login(String email, String password){
        Optional<User> userOpt = userRepository.findByEmail(email);
        if(userOpt.isPresent()){
            User user = userOpt.get();
            return passwordEncoder.matches(password, user.getPassword());
        }
        return false;
    }

    public boolean resetPassword(String username, String email, String newpassword){
        Optional<User> userOpt = userRepository.findByUsername(username);
        if(userOpt.isPresent()){
            User user = userOpt.get();
            if(user.getEmail().equals(email)){
                user.setPassword(passwordEncoder.encode(newpassword));
                userRepository.save(user);
                return true;
            }else{
                return false;
            }
        }else{
            return false;
        }
    }

    public List<User> getAllUsers(){
        return userRepository.findAll();
    }
    public User createUser(User user){
        userValidator.validateEmail(user.getEmail(), null);
        userValidator.validatePassword(user.getPassword());
        userValidator.validatePhoneNumber(user.getPhoneNumber(), null);
        userValidator.validateUsername(user.getUsername(), null);
        if (user.getRole() != null && user.getRole().getId() != null) {
        Role role = roleRepository.findById(user.getRole().getId())
                        .orElseThrow(() -> new RuntimeException("Role không tồn tại"));
        user.setRole(role); 
        }
        user.setPassword(passwordEncoder.encode(user.getPassword()));
        if (user.isDelete() == null) user.setDelete(false); 
        if (user.getStatus() == null) user.setStatus("ACTIVE");
        return userRepository.save(user);
    }
    public User getUser(Long id){
    return userRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Không tìm thấy người dùng."));
    }

    public User updateUser(Long id, User userDetails) {
    User user = userRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Không tìm thấy người dùng."));

   
    if (userDetails.getUsername() != null && !userDetails.getUsername().isBlank()) {
        userValidator.validateUsername(userDetails.getUsername(), id);
        user.setUsername(userDetails.getUsername());
    }

    if (userDetails.getEmail() != null && !userDetails.getEmail().isBlank()) {
        userValidator.validateEmail(userDetails.getEmail(), id);
        user.setEmail(userDetails.getEmail());
    }

    if (userDetails.getPhoneNumber() != null && !userDetails.getPhoneNumber().isBlank()) {
        userValidator.validatePhoneNumber(userDetails.getPhoneNumber(), id);
        user.setPhoneNumber(userDetails.getPhoneNumber());
    }

    if (userDetails.getPassword() != null && !userDetails.getPassword().isBlank()) {
        userValidator.validatePassword(userDetails.getPassword());
        user.setPassword(passwordEncoder.encode(userDetails.getPassword()));
    }

    // Update các field khác
    if (userDetails.getAvaUrl() != null) user.setAvaUrl(userDetails.getAvaUrl());
    if (userDetails.getRole() != null && userDetails.getRole().getId() != null) {
        Role role = roleRepository.findById(userDetails.getRole().getId())
                .orElseThrow(() -> new RuntimeException("Role không tồn tại"));
        user.setRole(role);
    }
    if (userDetails.getStatus() != null) user.setStatus(userDetails.getStatus());
    if (userDetails.isDelete() != null) user.setDelete(userDetails.isDelete());

    return userRepository.save(user);
}




    public void deleteUser(Long id){
        User user = userRepository.findById(id).orElseThrow(() -> new RuntimeException("Tài khoản không tồn tại."));
        userRepository.deleteById(id);
    }

    public boolean addCartoUser(Long userId, Car car){
        return userRepository.findById(userId).map(user -> {
            car.setOwner(user);
            carRepository.save(car);
            return true;
        }).orElse(false);
    }
}
