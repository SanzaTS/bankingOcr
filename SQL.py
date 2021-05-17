import mysql.connector

# establishing the connection
conn = mysql.connector.connect(
    user='root', password=' ', host='127.0.0.1', database='bankOcr')

# Creating a cursor object using the cursor() method
cursor = conn.cursor()

# Preparing SQL query to INSERT a record into the database.

# ql = """INSERT INTO user(name,phone,Sign)
#      VALUES (%s, %s, %s)"""


# tuples = ('Junior', '0125', 'N')

id = '921109573908'
fname = 'Sbusiso'
sName = 'Edward'
lNmae = 'Sithole'
addr = '11 peach str '
city = 'Durban'
province = 'KZN'
code = '4000'
email = 'sbudda.s@gmail.com'
phone = '0117144431'
account_number= '02354785'
account_name= 'savings'

sql = """INSERT INTO customer(id, F_Name, S_Name, L_Name, Str_Name, City, province, code,email,phone) 
              VALUES (%s, %s, %s, %s, %s, %s ,%s ,%s,%s,%s)"""
tuples = (id, fname, sName, lNmae, addr, city, province, code, email, phone)
# Executing the SQL command
cursor.execute(sql, tuples)

sql2 = """INSERT INTO account(Account_Number, Name, idNum)
                  VALUES (%s, %s, %s) """
tuples2 = (account_number, account_name, id)

cursor.execute(sql2, tuples2)

# Commit your changes in the database
conn.commit()

# Closing the connection
conn.close()
