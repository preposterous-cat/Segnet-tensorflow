var scrollSpy = new bootstrap.ScrollSpy(document.body, {
    target: '#navbar'
  })

function previewFile(input){
  var file = $("input[type=file]").get(0).files[0];

  if(file){
      var reader = new FileReader();

      reader.onload = function(){
          $("#previewImg").attr("src", reader.result);
      }

      reader.readAsDataURL(file);
  }
}

var alertList = document.querySelectorAll('.alert')
var alerts =  [].slice.call(alertList).map(function (element) {
  return new bootstrap.Alert(element)
})

