# l=int(input("enter the values"))
# b=int(input("enter the values"))
# area=0.5*l*b
# a=int(area)
# print("area",a)
#
#
# rev=0
# while(a!=0):
#     rem=a%10
#     rev=(rev*10)+rem
#     a=a//10
# print(rev)
#
# if area ==rev:
#     print("the area palindrom")
# else:
#     print("Not a palindrom")
#
# for i in range(1,11,2):
#     print(i)

pin=1234
tot_amt=50000

while True:
    password=int(input("Enter the password "))
    if password==pin:
        print("1.Withdraw\t2.Deposit\t3.Check Balance\t4.Change password")
        ch=int(input("Enter the choice"))
        if(ch==1):
            amt=int(input("Enter the amount to withdraw"))
            tot_amt=tot_amt-amt
            print("remaining balance is:",tot_amt)

        if(ch==2):
            amt = int(input("Enter the amount to deposite"))
            tot_amt = tot_amt + amt
            print(" balance is:", tot_amt)

        if(ch==3):
            print("Balance is:",tot_amt)

        if(ch==4):
            old_pin=int(input("enter the old pin in 4 digit"))
            if old_pin==pin:
                pin=int(input("Enter the  new pin in 4 digit"))
            else:
                print("Enter the valid old pin")
    else:
        print("Please enter the valid password!!!!!")
