from flask import Flask, render_template, request
import numpy as np
import model as mdl

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def home():
    output = 0
    answer = 0
    if request.method == 'POST':
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        self_employed = request.form['self_employed']
        credit_history = request.form['credit_history']        
        property_area = request.form['property_area']
        applicantincome = request.form['applicantincome']
        coapplicantincome = request.form['coapplicantincome']
        loan_amount = request.form['loan_amount']
        #LoanAmount_log = np.log(loan_amount)
        #TotalIncome = applicantincome + coapplicantincome
        #TotalIncome_log = np.log(TotalIncome)

        list1 = []
        list1.append(1) if gender == 'Male' else list1.append(0)
        list1.append(1) if married == 'Yes' else list1.append(0) 
        list1.append(int(dependents))
        list1.append(1) if education == 'Graduate' else list1.append(0)          
        list1.append(1) if self_employed == 'Yes' else list1.append(0)
        list1.append(360.0)
        list1.append(int(credit_history))
        list1.append(0) if property_area == 'Rural' else list1.append(1) if property_area == 'Semi-Urban' else list1.append(2)
        list1.append(int(applicantincome))
        list1.append(int(coapplicantincome))
        list1.append(int(loan_amount))

        output = mdl.predict_Loan_Approval(list1)
        answer = 'Loan Approved' if output == 1 else 'Loan Rejected'

    return render_template('index.html', result = answer)

if __name__ == "__main__":
    app.run(debug=True) 