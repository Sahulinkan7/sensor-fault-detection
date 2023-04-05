var upload=document.querySelector('button')

upload.onclick=(e)=>{
    var file_name = document.querySelector('input').value
    var file_format=file_name.slice(-4)
    error=[]
    if (file_name.length<=0 ){
        e.preventDefault()
        alert("File must be selected")
    }
    else{
        if (file_format!=".csv")
        {
            e.preventDefault()
            alert("file must be in .csv format")
        }
    }
    
}