#define PI 3.1415926535897932384626433832795 
#include <QMC5883L.h>
#include <Wire.h>

QMC5883L compass;

int N = 40;
float diameter = 6; //diameter in cm

int enco_L = 2,enco_R = 3;   //encoder input pins

//motor pins
int right_1 = 7 ,right_2 = 8;
int left_3 = 12 ,left_4 = 13;

//motor PWM pins
int right_pwm = 6 , left_pwm = 5;


//current coordinates
int uu = 0,vv = 0; 

//Destination coordinates
int xx ,yy ;

//Destination coordinates in the translated coordinates
int xx_new,yy_new;

//Distance
float dist_bw_cur_dest;

//Angle made
float angle_c;

int radio;

//---------------------------------------------------------------------------------------------------------------------------------------------------------
volatile int c_L = 0,c_R = 0;

// pwm speed
int motor_speed = 100;
int motor_kp , motor_com=0;

//diff bw heading and dest
int error;
float kp = 50.0/360.0 ;



 unsigned long num_ticks_l;
  unsigned long num_ticks_r;



  // Used to determine which way to turn to adjust
  unsigned long diff_l;
  unsigned long diff_r;


  // Remember previous encoder counts
  unsigned long enc_l_prev = c_L;
  unsigned long enc_r_prev = c_R;
//---------------------------------------------------------------------------------------------------------------------------------------------------------

float dist_L = 0;
int Ltick = 0;
int Ltick_prev = 0;
int delta_Ltick = 0;

float dist_R = 0;
int Rtick = 0;
int Rtick_prev = 0;
int delta_Rtick = 0;

float dist_c = 0 ;  //distance moved by the central point
float phi = 0;     //initial angular position

float v_lin = 0;           //required PWM 

float error_s = 0;                                        // error 
float Kp_s = 2.0;                                          //proportion control                                
int PWM_r = 0;                                           // PWM value to set
int PWM_l = 0; 

int done;
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------

void setup()
{

  Serial.begin(9600);
  
   //right motor
pinMode(right_1,OUTPUT);
pinMode(right_2,OUTPUT);
pinMode(right_pwm,OUTPUT);

 //left motor
pinMode(left_3,OUTPUT);
pinMode(left_4,OUTPUT);
pinMode(left_pwm,OUTPUT);

pinMode(enco_L,INPUT); //Left
pinMode(enco_R,INPUT); //Right
 
  
attachInterrupt(digitalPinToInterrupt(enco_L),count_L,CHANGE);
attachInterrupt(digitalPinToInterrupt(enco_R),count_R,CHANGE);

  Wire.begin();

  compass.init();
  compass.setSamplingRate(50);

  
  

  calibrate();        // calibration of compass-------------------------------------------------------

  delay(1000);
  
  
}


void calibrate(){

  while(Serial.available() == 0){

    int heading = compass.readHeading();
   
   heading -=74; //  74 is the angle made by our coordinate system(y axis) with the the magnetic NORTH in clockwise direction
  if((-74 <= heading ) && (heading <= 0)){
       heading += 360;
       
  }
    //Serial.print("heading : ");
    Serial.println(heading);

  
  }
  radio = Serial.parseInt();        //unwanted
}
void node_turning(int dest_angle){

  c_R = 0;
  c_L = 0;

  
  while(true)
  {
  
   int heading = compass.readHeading();
  heading -=74;  //  74 is the angle made by our coordinate system(y axis) with the the magnetic NORTH in clockwise direction
  if((-74 <= heading ) && (heading <= 0)){
       heading += 360;
       
  }
 

  if((0 <= dest_angle) && (dest_angle >= 180)&&((heading == 0) || (heading == 360))){
    heading = 0; 
  }
  if((180 < dest_angle) && (dest_angle >= 360)&&((heading == 0) || (heading == 360))){
    heading = 360;
 }
    Serial.print(" heading : ");
    Serial.println(heading);

 //error based on encoder
 
  // Sample number of encoder ticks
    num_ticks_l = c_L;
    num_ticks_r = c_R;


    // Number of ticks counted since last time
    diff_l = num_ticks_l - enc_l_prev;
    diff_r = num_ticks_r - enc_r_prev;

    // Store current tick counter for next time
    enc_l_prev = num_ticks_l;
    enc_r_prev = num_ticks_r;

   motor_com = (diff_r - diff_l)*(90.0/140.0);
     
   

Serial.print("motor_com : ");
Serial.println(motor_com);
    
 //error based on magnetometer
 
  error = dest_angle - heading;

  if(error > 0)
  {
    if(error > 180){
    error = 360 - dest_angle + heading;
    motor_kp = motor_speed + (kp*error);
    //motor_kp = -(kp*error);
      //motor_kp = constrain(motor_kp,0,200);
      //motor_kp = 150;
    
    analogWrite(left_pwm, constrain((motor_kp + motor_com),50,150));
    analogWrite(right_pwm,constrain((motor_kp - motor_com),50,150));

      digitalWrite(right_1,HIGH);
      digitalWrite(right_2,LOW);  //RIGHT MOTOR FORWARD    //LEFT TURN

      digitalWrite(left_3,LOW);
      digitalWrite(left_4,HIGH);   //LEFT MOTOR BACKWARD

      Serial.println(motor_kp);
      Serial.println(constrain((motor_kp + motor_com),50,150));
      Serial.println(constrain((motor_kp - motor_com),50,150));
    
      
    }
    else
    {
    motor_kp = motor_speed + (kp*error);
    //motor_kp = (kp*error);
    //motor_kp = constrain(motor_kp,0,255);
    //motor_kp = 150;
    
   analogWrite(left_pwm, constrain((motor_kp + motor_com),50,150));
   analogWrite(right_pwm,constrain((motor_kp - motor_com),50,150));

      digitalWrite(right_1,LOW);
      digitalWrite(right_2,HIGH);  //RIGHT MOTOR BACKWARD    //RIGHT TURN

      digitalWrite(left_3,HIGH);
      digitalWrite(left_4,LOW);   //LEFT MOTOR FORWARD

      Serial.println(motor_kp);
      Serial.println(constrain((motor_kp + motor_com),50,150));
      Serial.println(constrain((motor_kp - motor_com),50,150));
  }
  }

  if(error < 0)
  {
    error = -error;
    if(error > 180)
    {
      error = 360 - heading + dest_angle;      // error > 180
      motor_kp = motor_speed + (kp*error);
    //motor_kp = (kp*error);
    //motor_kp = constrain(motor_kp,0,255);
    //motor_kp = 150;
    
   analogWrite(left_pwm, constrain((motor_kp + motor_com),50,150));
   analogWrite(right_pwm,constrain((motor_kp - motor_com),50,150));

      digitalWrite(right_1,LOW);
      digitalWrite(right_2,HIGH);  //RIGHT MOTOR BACKWARD    //RIGHT TURN

      digitalWrite(left_3,HIGH);
      digitalWrite(left_4,LOW);   //LEFT MOTOR FORWARD

      Serial.println(motor_kp);
      Serial.println(constrain((motor_kp + motor_com),50,150));
      Serial.println(constrain((motor_kp - motor_com),50,150));
    }

    else{
      motor_kp = motor_speed +(kp*error);
      //motor_kp = -(kp*error);
      //motor_kp = constrain(motor_kp,0,200);
      //motor_kp = 150;
    
    analogWrite(left_pwm, constrain((motor_kp + motor_com),50,150));
    analogWrite(right_pwm,constrain((motor_kp - motor_com),50,150));

      digitalWrite(right_1,HIGH);
      digitalWrite(right_2,LOW);  //RIGHT MOTOR FORWARD    //LEFT TURN

      digitalWrite(left_3,LOW);
      digitalWrite(left_4,HIGH);   //LEFT MOTOR BACKWARD

      Serial.println(motor_kp);
      Serial.println(constrain((motor_kp + motor_com),50,150));
      Serial.println(constrain((motor_kp - motor_com),50,150));
  }
  }
  delay(5);

  if(((dest_angle - 2)<= heading )&&(heading <= (dest_angle + 2)))
  
  {
    break;
  }
  
  
}

brake();

}

void brake(){

     digitalWrite(right_1,LOW);
     digitalWrite(right_2,LOW);  //RIGHT MOTOR 

     digitalWrite(left_3,LOW);
     digitalWrite(left_4,LOW);   //LEFT MOTOR

     analogWrite(left_pwm, 0);
     analogWrite(right_pwm, 0);
}
void count_L()
{

  


  c_L++;
  
   //Serial.print("L : ");
   //Serial.println( c_L);
  Ltick++;

  
   //Serial.println(" ");
   //Serial.print("R : ");
   //Serial.println( Rtick);
  
}

void count_R()
{

  c_R++;
  
    //Serial.print("R : ");
    //Serial.println( c_R);
  Rtick++;
       
   //Serial.println(" ");
   //Serial.print("R : ");
   //Serial.println( Rtick);
  
}
void loop() {
 while(0 == 0){

  // destination coordinates  
  while(Serial.available() == 0)
  {   
  }
  xx = Serial.parseInt();
  delay(1000);
  while(Serial.available() == 0)
  {   
  }
  yy = Serial.parseInt();

  if((uu == xx) && (vv == yy)){
    
  }
 else{
  while(1 == 1){
  
Serial.println(uu);
Serial.println(vv);




 xx_new = xx - uu;
 Serial.println(xx_new);
 yy_new = yy - vv;
 Serial.println(yy_new);
 
//Distance claculation
 
 float vet_x = xx_new; //xx - uu;
 float vet_y = yy_new; //yy - vv;
 
  dist_bw_cur_dest = sqrt(((vet_x)*(vet_x)) + ((vet_y)*(vet_y)));
  //Serial.println("yes");
  Serial.println(dist_bw_cur_dest);


//Angle calculation clockwise with y axis
   
   
   angle_c =(atan(vet_y/vet_x))*(180/PI);  //(yy - vv)/(xx -uu) or y_new/x_new // radians to degree

 // when point is in 1st or 2nd quadrant 

   if(((xx_new > 0) && (yy_new > 0))||((xx_new > 0) && (yy_new < 0)))
   {
    angle_c = 90 - angle_c;          
   }
 // when point is in 3rd or 4th quadrant

   if(((xx_new < 0) && (yy_new < 0)) || ((xx_new < 0) && (yy_new > 0)))
   {
    angle_c = 90 - angle_c + 180;
   }
// when point is on y axis

   if((xx_new == 0) && (yy_new > 0)){
    angle_c = 0;
   }
   if((xx_new == 0) && (yy_new < 0)){
    angle_c = 180;
   }
// when point is on x axis

   if((xx_new > 0) && (yy_new == 0)){
    angle_c = 90; 
   }
   if((xx_new < 0) && (yy_new == 0)){
    angle_c = 270; 
   }
      
   Serial.println(angle_c);

//turning around the node------------------------------------------------------------------------------------------------------------------------------------------------
  // if error +ve right turn -ve left turn
 
  node_turning(angle_c);

  delay(1000);
//straight line movement along the destination angle---------------------------------------------------------------------------------------------------------------------

  straight_line_follow(angle_c,dist_bw_cur_dest);
  


 

 
   
   uu = xx;
   vv = yy;




break;

  
  
  
  }
 } 
}
}
void straight_line_follow(float phid,float distance_req){

  Ltick = 0;
  Rtick = 0;
  Ltick_prev = 0;
  Rtick_prev = 0;
  dist_c = 0;
  done = 0;
  
  while(3 == 3)
  {           
 

  delay(5); 
    
   int heading = compass.readHeading();
   
   heading -= 74;           //  74 is the angle made by our coordinate system(y axis) with the the magnetic NORTH in clockwise direction
  if((-74 <= heading ) && (heading <= 0)){
       heading += 360;
       
  }
 

  if((0 <= phid) && (phid >= 180)&&((heading == 0) || (heading == 360))){
    heading = 0; 
  }
  if((180 < phid) && (phid >= 360)&&((heading == 0) || (heading == 360))){
    heading = 360;
 }
  float phi = heading;
  Serial.print("heading : ");
   Serial.println(phi); 

   

    
//error calculation------------------------------------------------------------------------------------------------------
    if(((0 <= phid) && (phid <=90)) && ((270 <= phi) && (phi < 360))){
      error_s = 0 - phid -(360 - phi);
    }
    else if(((270 <= phid )&& (phid <= 360)) && ((0 < phi) && (phi < 90))){
      error_s = 360 + phi - phid;
    }
    else{  
    error_s = phi - phid;
    }
   
    v_lin = 100;                        //constant velocity --- req PWM
    
    PWM_r = v_lin + (Kp_s*error_s);
    PWM_l = v_lin - (Kp_s*error_s);

    PWM_l = constrain(PWM_l,50,150);
    PWM_r = constrain(PWM_r,50,150);

   Serial.print("error : ");
   Serial.println(error_s); 
    Serial.print("correction : ");
   Serial.println(error_s*Kp_s); 
   odometry(distance_req);

   if(done == 1){
    break;
   }
   digitalWrite(right_1,HIGH);
   digitalWrite(right_2,LOW);  //RIGHT MOTOR FORWARD

   digitalWrite(left_3,HIGH);
   digitalWrite(left_4,LOW);   //LEFT MOTOR FORWARD

   Serial.print(PWM_l); // to plot in serial plotter
   Serial.print(' ');
   Serial.println(PWM_r); // to plot in serial plotter

   


}
}

void odometry(float dest_dist){

   delta_Ltick = Ltick - Ltick_prev;
   dist_L = PI*diameter*(delta_Ltick/(double) N);

   delta_Rtick = Rtick - Rtick_prev;
   dist_R = PI*diameter*(delta_Rtick/(double) N);

   dist_c += (dist_L + dist_R)/2;
   
   Serial.print("dist_c : ");
   Serial.println(dist_c);

   if(dist_c >= dest_dist){
    brake();
    done = 1;
    Serial.println("done");
    }
    else{
    
   analogWrite(left_pwm,PWM_l);
   analogWrite(right_pwm,PWM_r);
    }

   Ltick_prev = Ltick;
   Rtick_prev = Rtick;
   
 
}
