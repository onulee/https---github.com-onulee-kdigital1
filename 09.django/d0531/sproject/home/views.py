from django.shortcuts import render
from product.models import Product

def index(request):
   qs = Product.objects.all()[:6]
   context = {'pList':qs}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
   
   return render(request,'index.html',context)
   
 
=======
   return render(request,'index.html',context)
   # return render(request,'index.html') 
>>>>>>> 18634d8d82627e0280ca5d1934434c06fa3ef464
=======
   return render(request,'index.html',context)
   # return render(request,'index.html') 
>>>>>>> 18634d8d82627e0280ca5d1934434c06fa3ef464
=======
   return render(request,'index.html',context)
   # return render(request,'index.html') 
>>>>>>> 18634d8d82627e0280ca5d1934434c06fa3ef464
